import difflib
import random
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
import ollama
import weaviate

from crag_utils import (
    extract_constraints,
    has_any_constraint,
    phone_satisfies,
)

WEAVIATE_HOST = "localhost"
WEAVIATE_HTTP_PORT = 8090
WEAVIATE_GRPC_PORT = 50051

COLLECTION_NAME = "MobilePhones"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

MODELS = [
    ("Qwen2.5", "qwen2.5:0.5b"),
    ("Llama3.2", "llama3.2:1b"),
    ("Gemma3", "gemma3:270m"),
]

# Order in which constraints get dropped during the "relax" stage of
# corrective retrieval. Softest / least-load-bearing constraints first.
RELAX_ORDER = ["min_year", "brand", "min_battery_mah", "min_ram_gb", "min_price", "max_price"]


embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)


client = weaviate.connect_to_local(
    host=WEAVIATE_HOST,
    port=WEAVIATE_HTTP_PORT,
    grpc_port=WEAVIATE_GRPC_PORT
)

collection = client.collections.get(COLLECTION_NAME)


def ensure_models_available():
    try:
        installed = {
            m["model"] if isinstance(m, dict) else m.model
            for m in ollama.list().get("models", [])
        }
    except Exception:
        installed = set()

    for display_name, model_name in MODELS:
        if model_name not in installed:
            print(f"[startup] Pulling missing model: {model_name} ({display_name})")
            try:
                ollama.pull(model_name)
            except Exception as e:
                print(f"[startup] Failed to pull {model_name}: {e}")


def ensure_collection_exists():
    if not client.collections.exists(COLLECTION_NAME):
        print(
            f"[startup] WARNING: collection '{COLLECTION_NAME}' does not exist yet. "
            f"Run ingest.py before calling /recommend, or every query will fail."
        )



@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_collection_exists()
    ensure_models_available()
    yield
    print("[shutdown] Closing Weaviate connection cleanly...")
    client.close()


app = FastAPI(
    title="Mobile Recommendation System",
    version="2.0-crag",
    lifespan=lifespan
)



def retrieve(query, n_results=20):

    query_embedding = embedding_model.encode(query)

    response = collection.query.near_vector(
        near_vector=query_embedding.tolist(),
        limit=n_results
    )

    return response.objects


def get_unique_phones(results, max_phones=5):

    phones = []
    seen_models = set()

    for obj in results:

        meta = obj.properties
        model_name = meta["model"]

        if model_name not in seen_models:
            seen_models.add(model_name)
            phones.append(meta)

        if len(phones) == max_phones:
            break

    return phones


# ---------------------------------------------------------------------------
# Corrective RAG layer
# ---------------------------------------------------------------------------
#
# Standard CRAG (Yan et al.) grades each retrieved doc as CORRECT / INCORRECT
# / AMBIGUOUS and then either refines the knowledge, discards it and falls
# back to an external source (web search, in the original paper), or both.
#
# Here there's no external corpus to fall back to, so "corrective retrieval"
# means: widen the vector search net, and if that still doesn't produce
# constraint-satisfying phones, progressively relax the weakest constraints
# rather than silently returning whatever the first query happened to find.
#
def corrective_retrieve(query, max_phones=5):
    """
    Returns (phones, correction_log).

    correction_log["correction_action"] is one of:
      - "none"               : query had no extractable constraints, behaves
                                exactly like the original pipeline
      - "correct_refine"     : most of the initial pool already satisfied the
                                constraints -> just filtered down to those
      - "ambiguous_combine"  : some satisfied, some didn't -> kept the
                                satisfying ones and merged in a wider retrieval
      - "incorrect_re_retrieve" : none satisfied -> widened retrieval found
                                   matches
      - "incorrect_relaxed"  : even the widened retrieval had nothing, so
                                constraints were progressively dropped
      - "fallback_no_match"  : nothing in the catalogue matches even relaxed
                                constraints -> falls back to plain vector
                                search results, with a warning surfaced
    """
    constraints = extract_constraints(query)

    correction_log = {
        "constraints_extracted": constraints,
        "correction_action": "none",
        "retrieval_rounds": 1,
        "relaxed_constraints": [],
        "warning": None,
    }

    # Stage 1: same retrieval as before, slightly larger pool to grade against.
    initial_results = retrieve(query, n_results=20)
    initial_phones = get_unique_phones(initial_results, max_phones=10)

    if not has_any_constraint(constraints):
        return initial_phones[:max_phones], correction_log

    if not initial_phones:
        correction_log["correction_action"] = "fallback_no_match"
        correction_log["warning"] = "No phones were retrieved at all for this query."
        return [], correction_log

    graded = [(p, phone_satisfies(p, constraints)) for p in initial_phones]
    satisfying = [p for p, (ok, _) in graded if ok]
    ratio = len(satisfying) / len(initial_phones)

    # CORRECT: most of what we retrieved already satisfies the constraints.
    # Refine = drop the ones that don't, recompose the context from only
    # the satisfying phones.
    if ratio >= 0.6:
        correction_log["correction_action"] = "correct_refine"
        return satisfying[:max_phones], correction_log

    action = "ambiguous_combine" if ratio > 0 else "incorrect_re_retrieve"
    correction_log["correction_action"] = action
    correction_log["retrieval_rounds"] = 2

    # Stage 2: widen the net considerably and re-grade.
    expanded_results = retrieve(query, n_results=100)
    expanded_phones = get_unique_phones(expanded_results, max_phones=40)

    graded_expanded = [(p, phone_satisfies(p, constraints)) for p in expanded_phones]
    satisfying_expanded = [p for p, (ok, _) in graded_expanded if ok]

    seen = set()
    merged = []
    for p in satisfying + satisfying_expanded:
        if p["model"] not in seen:
            seen.add(p["model"])
            merged.append(p)

    if merged:
        return merged[:max_phones], correction_log

    # Stage 3: AMBIGUOUS/INCORRECT and the wider pool still has nothing ->
    # progressively relax the softest constraints and retry against the
    # already-expanded pool (no need to hit Weaviate again).
    relaxed = dict(constraints)
    correction_log["retrieval_rounds"] = 3

    for key in RELAX_ORDER:
        if relaxed.get(key) is None:
            continue

        correction_log["relaxed_constraints"].append(key)
        relaxed[key] = None

        graded_relaxed = [(p, phone_satisfies(p, relaxed)) for p in expanded_phones]
        satisfying_relaxed = [p for p, (ok, _) in graded_relaxed if ok]

        if satisfying_relaxed:
            correction_log["correction_action"] = "incorrect_relaxed"
            return satisfying_relaxed[:max_phones], correction_log

    # Stage 4: nothing in the catalogue matches, even relaxed. Fall back to
    # the plain vector-search results rather than returning an empty list,
    # but be explicit that this is a fallback.
    correction_log["correction_action"] = "fallback_no_match"
    correction_log["warning"] = (
        "No phones in the catalogue matched the extracted constraints, "
        "even after relaxing them one by one. Showing closest vector-search "
        "results instead."
    )
    return initial_phones[:max_phones], correction_log


def ask_model(model_name, phones, query):
    shuffled_phones = phones.copy()
    rng = random.Random(f"{model_name}::{query}")
    rng.shuffle(shuffled_phones)

    context = ""
    phone_names = []

    for phone in shuffled_phones:

        phone_names.append(phone["model"])

        context += f"""
Company: {phone['company']}
Model: {phone['model']}
RAM: {phone['ram']}
Processor: {phone['processor']}
Battery: {phone['battery']}
Price: {phone['price']}
Launch Year: {phone['launch_year']}
"""

    options_list = "\n".join(f"- {name}" for name in phone_names)

    prompt = f"""You are a smartphone recommendation expert.

Example:
User Query: longest battery life
Available Phones:
- Galaxy M14
- iPhone 13
Answer: Galaxy M14

Now do the same for this real request.

User Query:
{query}

Available Phones:

{context}

You must choose EXACTLY ONE phone from this list of model names, copied exactly as written:

{options_list}

Answer with ONLY the model name on a single line. No explanation, no punctuation,
no extra words.
"""

    try:

        response = ollama.chat(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            options={
                "temperature": 0,
                "num_predict": 30
            }
        )

        raw_text = response["message"]["content"].strip()
        recommendation = raw_text.split("\n")[0].strip().strip('"').strip("'")

        if not recommendation:
            return "No Recommendation", "Empty response from model"

        return recommendation, None

    except Exception as e:
        traceback.print_exc()
        return "No Recommendation", str(e)

def get_phone_details(model_name, phones):
    """
    Returns (details_dict, match_type).
    match_type is one of: "exact", "fuzzy", "fallback", "none"

    "fallback" means the model's raw output did not match any retrieved
    phone — the entry shown is just the top vector-search result, NOT the
    model's actual choice. Keep an eye on this field; if a model shows
    "fallback" frequently, it's not really doing the recommendation task.
    """

    # 1. Exact / substring match first (handles the easy case)
    for phone in phones:
        if model_name.lower() in phone["model"].lower() \
                or phone["model"].lower() in model_name.lower():
            return dict(phone), "exact"

    # 2. Fuzzy match — catches cases where the LLM slightly rewords
    #    the model name (e.g. adds "5G", drops a hyphen, etc.)
    candidate_names = [phone["model"] for phone in phones]
    close = difflib.get_close_matches(model_name, candidate_names, n=1, cutoff=0.4)

    if close:
        matched_name = close[0]
        for phone in phones:
            if phone["model"] == matched_name:
                return dict(phone), "fuzzy"

    # 3. Fallback — never silently return "Unknown" for everything.
    if phones:
        return dict(phones[0]), "fallback"

    return {
        "company": "Unknown",
        "model": model_name,
        "ram": "Unknown",
        "battery": "Unknown",
        "processor": "Unknown",
        "launch_year": "Unknown",
        "price": "Unknown"
    }, "none"


@app.get("/")
def home():

    total_records = collection.aggregate.over_all(total_count=True).total_count

    return {
        "project": "Mobile Recommendation System",
        "vector_database": "Weaviate",
        "embedding_model": EMBEDDING_MODEL_NAME,
        "llms": [display_name for display_name, _ in MODELS],
        "total_records": total_records,
        "retrieval_strategy": "Corrective RAG (constraint grading + adaptive re-retrieval)"
    }



@app.get("/recommend")
def recommend(query: str):

    phones, correction_log = corrective_retrieve(query)

    recommendations = []

    if not phones:
        return {
            "user_query": query,
            "corrective_rag_details": correction_log,
            "top_recommendation": None,
            "other_model_recommendations": [],
            "error": "No phones survived retrieval/correction for this query."
        }

    for display_name, model_name in MODELS:

        raw_recommendation, error = ask_model(model_name, phones, query)
        phone_details, match_type = get_phone_details(raw_recommendation, phones)

        entry = {
            "recommended_by": display_name,
            "model": phone_details.get("model"),
            "company": phone_details.get("company"),
            "ram": phone_details.get("ram"),
            "battery": phone_details.get("battery"),
            "processor": phone_details.get("processor"),
            "launch_year": phone_details.get("launch_year"),
            "price": phone_details.get("price"),
            "raw_model_output": raw_recommendation,
            "match_type": match_type,
        }

        if error:
            entry["error"] = error

        recommendations.append(entry)

    top_recommendation = recommendations[0]
    other_recommendations = recommendations[1:]

    total_records = collection.aggregate.over_all(total_count=True).total_count

    return {
        "user_query": query,
        "chunking_details": {
            "chunking_method": "Row Based Chunking",
            "total_records": total_records,
            "retrieved_chunks": len(phones)
        },
        "corrective_rag_details": correction_log,
        "top_recommendation": top_recommendation,
        "other_model_recommendations": other_recommendations
    }