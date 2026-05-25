from __future__ import annotations

import json
import os
import re
import tempfile
import urllib.parse
import urllib.request
from functools import lru_cache
from pathlib import Path
from threading import RLock
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from mixedbread import Mixedbread


ROOT = Path(__file__).resolve().parent
DEFAULT_STORE_FILE = ROOT / "mixedbread_store.json"
FIELD_DATA_FILE = ROOT / "data" / "learning-to-rank.json"
DYNAMIC_CACHE_FILE = ROOT / ".scholartrace_dynamic.json"
CACHE_VERSION = 6
SEMANTIC_SCHOLAR_BASE = "https://api.semanticscholar.org/graph/v1"
OPENALEX_BASE = "https://api.openalex.org"
DYNAMIC_CACHE_LOCK = RLock()
AVATAR_COLORS = [
    "amber",
    "rose",
    "purple",
    "blue",
    "indigo",
    "emerald",
    "fuchsia",
    "sky",
    "stone",
    "violet",
    "slate",
]
PERSON_QUERY_ALIASES = {
    "feifei": ["Fei-Fei Li", "Li Fei-Fei", "Fei Fei Li"],
    "hinton": ["Geoff Hinton", "Geoffrey Hinton", "Geoffrey E. Hinton"],
}
PERSON_QUERY_ALIAS_FILTERS = {
    "feifei": {"fei"},
    "hinton": {"geoff", "geoffrey"},
}
PERSON_QUERY_REQUIRED_TOKENS = {
    "feifei": {"li"},
    "hinton": {"hinton"},
}
OPENALEX_PERSON_OVERRIDES = {
    "feifei": {
        "openalex_id": "A5100450462",
        "name": "Fei-Fei Li",
        "institution": "Stanford University",
    }
}

SOURCE_TO_SCHOLAR_ID = {
    "christopher_burges.md": "christopher-burges",
    "hang_li.md": "hang-li",
    "hongyuan_zha.md": "hongyuan-zha",
    "jun_xu.md": "jun-xu",
    "olivier_chapelle.md": "olivier-chapelle",
    "stephen_robertson.md": "stephen-robertson",
    "tao_qin.md": "tao-qin",
    "thorsten_joachims.md": "thorsten-joachims",
    "tie_yan_liu.md": "tie-yan-liu",
    "w_bruce_croft.md": "bruce-croft",
}


app = FastAPI(title="ScholarTrace RAG API")
app.mount("/data", StaticFiles(directory=ROOT / "data"), name="data")


def object_field(value: Any, field: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(field, default)
    return getattr(value, field, default)


def to_plain_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return {
        key: getattr(value, key)
        for key in dir(value)
        if not key.startswith("_") and not callable(getattr(value, key))
    }


def query_key(value: str) -> str:
    return " ".join(value.lower().split())


def object_metadata(item: dict[str, Any]) -> dict[str, Any]:
    metadata = item.get("metadata") or {}
    return metadata if isinstance(metadata, dict) else {}


def scholar_id_from_author_id(author_id: str) -> str:
    value = str(author_id)
    if value.startswith("oa-"):
        return value
    return f"ss-{value}"


@lru_cache(maxsize=1)
def load_client() -> Mixedbread:
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("MXBAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="MXBAI_API_KEY is missing")
    return Mixedbread(api_key=api_key)


@lru_cache(maxsize=1)
def load_store_identifier() -> str:
    configured = Path(os.getenv("MIXEDBREAD_STORE_FILE", DEFAULT_STORE_FILE))
    store_file = configured if configured.is_absolute() else ROOT / configured
    if not store_file.exists():
        raise HTTPException(status_code=500, detail=f"Store file not found: {store_file}")
    payload = json.loads(store_file.read_text(encoding="utf-8"))
    store_identifier = payload.get("store_id") or payload.get("store_name")
    if not store_identifier:
        raise HTTPException(status_code=500, detail="Store id/name missing from store file")
    return store_identifier


@lru_cache(maxsize=1)
def load_static_field() -> dict[str, Any]:
    if not FIELD_DATA_FILE.exists():
        return {"scholars": [], "edges": []}
    return json.loads(FIELD_DATA_FILE.read_text(encoding="utf-8"))


def default_dynamic_cache() -> dict[str, Any]:
    return {"version": CACHE_VERSION, "queries": {}, "scholars": {}, "edges": []}


def dynamic_scholar_limit() -> int:
    load_dotenv(ROOT / ".env")
    try:
        return max(1, min(20, int(os.getenv("DYNAMIC_SCHOLAR_LIMIT", "5"))))
    except ValueError:
        return 5


def load_dynamic_cache() -> dict[str, Any]:
    with DYNAMIC_CACHE_LOCK:
        if not DYNAMIC_CACHE_FILE.exists():
            return default_dynamic_cache()
        try:
            payload = json.loads(DYNAMIC_CACHE_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return default_dynamic_cache()
        if payload.get("version") != CACHE_VERSION:
            return default_dynamic_cache()
        payload.setdefault("version", CACHE_VERSION)
        payload.setdefault("queries", {})
        payload.setdefault("scholars", {})
        payload.setdefault("edges", [])
        return payload


def save_dynamic_cache(cache: dict[str, Any]) -> None:
    with DYNAMIC_CACHE_LOCK:
        tmp_file = DYNAMIC_CACHE_FILE.with_suffix(".tmp")
        tmp_file.write_text(
            json.dumps(cache, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp_file.replace(DYNAMIC_CACHE_FILE)


def static_semantic_author_ids() -> set[str]:
    ids = set()
    for scholar in load_static_field().get("scholars", []):
        metrics = scholar.get("metrics") or {}
        for key in ("semantic_scholar_id", "semantic_scholar_id_alt"):
            value = metrics.get(key)
            if isinstance(value, list):
                ids.update(str(item) for item in value if item)
            elif value:
                ids.add(str(value))
    return ids


def scholar_by_id(scholar_id: str) -> dict[str, Any] | None:
    for scholar in load_static_field().get("scholars", []):
        if scholar.get("id") == scholar_id:
            return scholar
    return load_dynamic_cache().get("scholars", {}).get(scholar_id)


def dynamic_payload_for_query(q: str) -> dict[str, Any]:
    cache = load_dynamic_cache()
    ids = cache.get("queries", {}).get(query_key(q), [])
    scholars = [
        cache.get("scholars", {}).get(scholar_id)
        for scholar_id in ids
        if cache.get("scholars", {}).get(scholar_id)
    ]
    id_set = {scholar["id"] for scholar in scholars}
    edges = [
        edge
        for edge in cache.get("edges", [])
        if edge.get("source") in id_set or edge.get("target") in id_set
    ]
    return {"scholars": scholars, "edges": edges}


def result_text(item: dict[str, Any]) -> str:
    for field in ("text", "summary", "ocr_text"):
        value = item.get(field)
        if value:
            return " ".join(str(value).split())
    return ""


def result_source(item: dict[str, Any]) -> str | None:
    metadata = object_metadata(item)
    source = metadata.get("source")
    if source:
        return str(source)
    filename = item.get("filename")
    if filename:
        return str(filename)
    return None


def normalize_result(rank: int, item: Any) -> dict[str, Any] | None:
    data = to_plain_dict(item)
    metadata = object_metadata(data)
    source = result_source(data)
    scholar_id = metadata.get("scholar_id") or SOURCE_TO_SCHOLAR_ID.get(source or "")
    if not scholar_id:
        return None
    text = result_text(data)
    return {
        "scholar_id": scholar_id,
        "rag_rank": rank,
        "rag_score": data.get("score"),
        "source": source,
        "snippet": text[:360],
        "dynamic": bool(metadata.get("dynamic")),
    }


def semantic_scholar_json(path: str, params: dict[str, Any]) -> dict[str, Any]:
    load_dotenv(ROOT / ".env")
    url = f"{SEMANTIC_SCHOLAR_BASE}{path}?{urllib.parse.urlencode(params)}"
    headers = {
        "User-Agent": "ScholarTrace/0.1 (+https://github.com/kangkangzi2025/scholartrace)",
    }
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=12) as response:
        return json.loads(response.read().decode("utf-8"))


def openalex_json(path: str, params: dict[str, Any]) -> dict[str, Any]:
    url = f"{OPENALEX_BASE}{path}?{urllib.parse.urlencode(params)}"
    headers = {
        "User-Agent": "ScholarTrace/0.1 (+https://github.com/kangkangzi2025/scholartrace)",
    }
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=12) as response:
        return json.loads(response.read().decode("utf-8"))


def paper_summary(paper: dict[str, Any]) -> dict[str, Any]:
    return {
        "paperId": paper.get("paperId"),
        "title": paper.get("title") or "Untitled paper",
        "venue": paper.get("venue") or "",
        "year": paper.get("year") or "",
        "citations": paper.get("citationCount") or 0,
        "abstract": paper.get("abstract") or "",
        "authors": [
            {
                "authorId": str(author.get("authorId")),
                "name": author.get("name") or "Unknown Author",
            }
            for author in paper.get("authors") or []
            if author.get("authorId")
        ],
    }


def openalex_author_id(raw_id: str) -> str:
    return "oa-" + str(raw_id).rstrip("/").split("/")[-1]


def openalex_work_summary(work: dict[str, Any]) -> dict[str, Any]:
    venue = ""
    location = work.get("primary_location") or {}
    source = location.get("source") or {}
    if isinstance(source, dict):
        venue = source.get("display_name") or ""
    return {
        "paperId": work.get("id"),
        "title": work.get("title") or "Untitled paper",
        "venue": venue,
        "year": work.get("publication_year") or "",
        "citations": work.get("cited_by_count") or 0,
        "abstract": "",
        "authors": [
            {
                "authorId": openalex_author_id((authorship.get("author") or {}).get("id") or ""),
                "name": (authorship.get("author") or {}).get("display_name") or "Unknown Author",
            }
            for authorship in work.get("authorships") or []
            if (authorship.get("author") or {}).get("id")
        ],
    }


def looks_like_person_query(q: str) -> bool:
    tokens = [token for token in re.split(r"\s+", q.strip()) if token]
    if not tokens or len(tokens) > 4:
        return False
    normalized = {re.sub(r"[^a-z]", "", token.lower()) for token in tokens}
    topic_words = {
        "ai",
        "bm",
        "diffusion",
        "graph",
        "language",
        "learning",
        "model",
        "models",
        "network",
        "networks",
        "neural",
        "rank",
        "ranking",
        "reinforcement",
        "retrieval",
        "search",
        "transformer",
        "vision",
    }
    if normalized & topic_words:
        return False
    if len(tokens) == 1:
        return bool(re.fullmatch(r"[A-Za-z][A-Za-z'.-]+", tokens[0]))
    return all(re.fullmatch(r"[A-Za-z][A-Za-z'.-]+", token) for token in tokens)


def fetch_author_papers(author_id: str) -> list[dict[str, Any]]:
    fields = "paperId,title,venue,year,citationCount,authors"
    try:
        payload = semantic_scholar_json(
            f"/author/{author_id}/papers",
            {"limit": 20, "fields": fields},
        )
    except Exception:
        return []
    return [paper_summary(paper) for paper in payload.get("data") or []]


def fetch_openalex_author_candidate(q: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]] | None:
    override = OPENALEX_PERSON_OVERRIDES.get(query_key(q))
    if not override:
        return None
    openalex_id = override["openalex_id"]
    author_url = f"https://openalex.org/{openalex_id}"
    try:
        author = openalex_json(
            f"/authors/{openalex_id}",
            {"select": "id,display_name,cited_by_count,works_count,summary_stats"},
        )
        works_payload = openalex_json(
            "/works",
            {
                "filter": f"authorships.author.id:{author_url}",
                "per-page": 20,
                "sort": "cited_by_count:desc",
                "select": "id,title,publication_year,cited_by_count,primary_location,authorships",
            },
        )
    except Exception:
        return None

    papers = [openalex_work_summary(work) for work in works_payload.get("results") or []]
    details = {
        "authorId": openalex_author_id(openalex_id),
        "name": override.get("name") or author.get("display_name") or "Unknown Author",
        "url": author_url,
        "affiliations": [override.get("institution") or "Unknown institution"],
        "hIndex": (author.get("summary_stats") or {}).get("h_index") or 0,
        "citationCount": author.get("cited_by_count") or 0,
        "paperCount": author.get("works_count") or 0,
        "source": "OpenAlex",
    }
    return [
        {
            "author_id": details["authorId"],
            "name": details["name"],
            "score": author_candidate_score(q, details),
            "papers": papers,
            "details": details,
            "role": f"OpenAlex 实时发现 · 与「{q}」匹配的学者",
        }
    ], papers


def person_search_queries(q: str) -> list[str]:
    key = query_key(q)
    aliases = PERSON_QUERY_ALIASES.get(key, [])
    queries = aliases or [q]
    deduped = []
    seen = set()
    for query in queries:
        normalized = query_key(query)
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(query)
    return deduped


def author_candidate_score(query: str, author: dict[str, Any]) -> float:
    name = query_key(author.get("name") or "")
    key = query_key(query)
    score = float(author.get("citationCount") or 0)
    score += float(author.get("hIndex") or 0) * 1000
    score += float(author.get("paperCount") or 0) * 10
    if key == name:
        score += 100_000
    elif all(token in name for token in key.split()):
        score += 25_000
    return score


def author_name_key(name: str) -> str:
    tokens = re.findall(r"[a-z]+", name.lower())
    if len(tokens) >= 2 and len(tokens[0]) > 1:
        return f"{tokens[0]} {tokens[-1]}"
    return " ".join(tokens)


def passes_person_alias_filter(q: str, name: str) -> bool:
    allowed_prefixes = PERSON_QUERY_ALIAS_FILTERS.get(query_key(q))
    if not allowed_prefixes:
        return True
    tokens = re.findall(r"[a-z]+", name.lower())
    required_tokens = PERSON_QUERY_REQUIRED_TOKENS.get(query_key(q), set())
    if required_tokens and not required_tokens.issubset(set(tokens)):
        return False
    return any(
        any(token.startswith(prefix) for prefix in allowed_prefixes)
        for token in tokens
    )


def add_coauthor_candidates(
    candidates: list[dict[str, Any]],
    papers: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    selected_author_ids = {candidate["author_id"] for candidate in candidates}
    buckets: dict[str, dict[str, Any]] = {}
    for paper in papers:
        for author in paper.get("authors") or []:
            author_id = str(author.get("authorId"))
            if not author_id or author_id in selected_author_ids:
                continue
            bucket = buckets.setdefault(
                author_id,
                {
                    "author_id": author_id,
                    "name": author.get("name") or "Unknown Author",
                    "score": 0.0,
                    "papers": [],
                    "details": {
                        "authorId": author_id,
                        "name": author.get("name") or "Unknown Author",
                    },
                    "role": "实时共作者节点",
                },
            )
            bucket["score"] += 1 + min(paper.get("citations") or 0, 5000) ** 0.5 / 50
            bucket["papers"].append(paper)
    coauthors = sorted(buckets.values(), key=lambda item: item["score"], reverse=True)
    return candidates + coauthors[: max(0, limit - len(candidates))]


def discover_author_search_candidates(q: str, limit: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    openalex_override = fetch_openalex_author_candidate(q)
    if openalex_override:
        primary_candidates, papers = openalex_override
        return add_coauthor_candidates(primary_candidates[:1], papers, limit), papers

    fields = "name,url,affiliations,hIndex,citationCount,paperCount"
    static_ids = static_semantic_author_ids()
    candidates_by_id: dict[str, dict[str, Any]] = {}
    search_limit = max(20, limit * 4)
    for search_query in person_search_queries(q):
        payload = semantic_scholar_json(
            "/author/search",
            {"query": search_query, "limit": search_limit, "fields": fields},
        )
        for author in payload.get("data") or []:
            author_id = author.get("authorId")
            if not author_id or str(author_id) in static_ids:
                continue
            if not passes_person_alias_filter(q, author.get("name") or ""):
                continue
            author_id = str(author_id)
            score = author_candidate_score(search_query, author)
            existing = candidates_by_id.get(author_id)
            if existing and existing["score"] >= score:
                continue
            candidates_by_id[author_id] = {
                "author_id": author_id,
                "name": author.get("name") or "Unknown Author",
                "score": score,
                "papers": [],
                "details": author,
            }

    deduped_by_name: dict[str, dict[str, Any]] = {}
    for candidate in sorted(candidates_by_id.values(), key=lambda item: item["score"], reverse=True):
        name_key = author_name_key(candidate["name"])
        if not name_key:
            continue
        if name_key not in deduped_by_name:
            deduped_by_name[name_key] = candidate

    primary_candidates = list(deduped_by_name.values())[:1]
    all_papers = []
    for candidate in primary_candidates:
        papers = fetch_author_papers(candidate["author_id"])
        all_papers.extend(papers)
        candidate["papers"] = papers
    return add_coauthor_candidates(primary_candidates, all_papers, limit), all_papers


def discover_author_candidates(q: str, limit: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if looks_like_person_query(q):
        return discover_author_search_candidates(q, limit)

    fields = "paperId,title,abstract,venue,year,citationCount,authors"
    payload = semantic_scholar_json(
        "/paper/search",
        {"query": q, "limit": 20, "fields": fields},
    )
    papers = payload.get("data") or []
    buckets: dict[str, dict[str, Any]] = {}

    for paper in papers:
        for author in paper.get("authors") or []:
            author_id = author.get("authorId")
            if not author_id:
                continue
            bucket = buckets.setdefault(
                str(author_id),
                {
                    "author_id": str(author_id),
                    "name": author.get("name") or "Unknown Author",
                    "score": 0.0,
                    "papers": [],
                },
            )
            bucket["score"] += 1 + min(paper.get("citationCount") or 0, 5000) ** 0.5 / 50
            bucket["papers"].append(paper_summary(paper))

    static_ids = static_semantic_author_ids()
    candidates = [
        value
        for value in buckets.values()
        if value["author_id"] not in static_ids
    ]
    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates[:limit], papers


def fetch_author_details(author_id: str) -> dict[str, Any]:
    fields = ",".join(
        [
            "authorId",
            "name",
            "url",
            "affiliations",
            "hIndex",
            "citationCount",
            "paperCount",
        ]
    )
    try:
        return semantic_scholar_json(f"/author/{author_id}", {"fields": fields})
    except Exception:
        return {}


def color_for_id(value: str) -> str:
    return AVATAR_COLORS[sum(ord(ch) for ch in value) % len(AVATAR_COLORS)]


def materialize_scholar(q: str, candidate: dict[str, Any]) -> dict[str, Any]:
    author_id = candidate["author_id"]
    details = candidate.get("details") or fetch_author_details(author_id)
    name = details.get("name") or candidate.get("name") or "Unknown Author"
    papers = sorted(
        candidate.get("papers") or [],
        key=lambda paper: paper.get("citations") or 0,
        reverse=True,
    )[:5]
    current_year = 2026
    recent_citations = sum(
        int(paper.get("citations") or 0)
        for paper in papers
        if isinstance(paper.get("year"), int) and paper["year"] >= current_year - 3
    )
    affiliations = details.get("affiliations") or []
    institution = affiliations[0] if affiliations else "Unknown institution"
    scholar_id = scholar_id_from_author_id(author_id)
    source_label = details.get("source") or "Semantic Scholar"
    role = candidate.get("role") or f"{source_label} 实时发现 · 与「{q}」相关的论文作者"
    field_papers = [
        {
            "title": paper["title"],
            "venue": paper.get("venue") or "Unknown venue",
            "year": paper.get("year") or "",
            "citations": paper.get("citations") or 0,
        }
        for paper in papers
    ]
    return {
        "id": scholar_id,
        "name": name,
        "chinese_name": "",
        "institution": institution,
        "country": "",
        "avatar_color": color_for_id(author_id),
        "role_in_field": role,
        "metrics": {
            "h_index": details.get("hIndex") or 0,
            "total_citations": details.get("citationCount") or 0,
            "recent_3y_citations": recent_citations,
            "field_papers": len(field_papers),
            "centrality": min(0.95, 0.45 + min(candidate.get("score", 0), 8) / 16),
            "consistency": 0.55,
            "semantic_scholar_id": author_id,
            "paper_count_ss": details.get("paperCount") or 0,
            "_realtime": True,
        },
        "field_papers": field_papers,
        "research_line": f"实时从 Semantic Scholar 以「{q}」检索到该学者；代表作按当前检索结果中的引用数排序。",
        "recommended_reading": [
            {
                "order": index + 1,
                "title": paper["title"],
                "why": "当前 query 命中的高引用论文",
                "difficulty": "★★",
            }
            for index, paper in enumerate(field_papers[:3])
        ],
        "llm_blurb": f"{name} 是本次「{q}」实时检索中发现的候选学者，相关性来自 Semantic Scholar 论文命中与 Mixedbread profile 检索。",
        "relation_to_center": "coauthor",
        "homepage_url": details.get("url") or f"https://www.semanticscholar.org/author/{author_id}",
        "scholar_url": details.get("url") or f"https://www.semanticscholar.org/author/{author_id}",
        "_dynamic": True,
        "_source": f"dynamic/{scholar_id}.md",
    }


def dynamic_edges_from_papers(selected_ids: set[str], papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    edge_map: dict[tuple[str, str], dict[str, Any]] = {}
    for paper in papers:
        author_ids = [
            scholar_id_from_author_id(str(author.get("authorId")))
            for author in paper.get("authors") or []
            if author.get("authorId") and scholar_id_from_author_id(str(author.get("authorId"))) in selected_ids
        ]
        for index, source in enumerate(author_ids):
            for target in author_ids[index + 1 :]:
                key = tuple(sorted((source, target)))
                edge = edge_map.setdefault(
                    key,
                    {
                        "source": key[0],
                        "target": key[1],
                        "type": "coauthor",
                        "weight": 0,
                        "evidence_titles": [],
                    },
                )
                edge["weight"] += 1
                title = paper.get("title")
                if title and len(edge["evidence_titles"]) < 3:
                    edge["evidence_titles"].append(title)
    edges = []
    for edge in edge_map.values():
        titles = "；".join(edge.pop("evidence_titles"))
        edge["evidence"] = f"实时发现共著 {edge['weight']} 篇匹配论文" + (f"：{titles}" if titles else "")
        edges.append(edge)
    return edges


def scholar_markdown(scholar: dict[str, Any]) -> str:
    metrics = scholar.get("metrics") or {}
    lines = [
        f"# {scholar.get('name')}",
        "",
        f"- ScholarTrace ID: {scholar.get('id')}",
        f"- Institution: {scholar.get('institution')}",
        f"- Role: {scholar.get('role_in_field')}",
        f"- h-index: {metrics.get('h_index', 0)}",
        f"- Total citations: {metrics.get('total_citations', 0)}",
        f"- Semantic Scholar ID: {metrics.get('semantic_scholar_id', '')}",
        "",
        "## Research line",
        scholar.get("research_line", ""),
        "",
        "## Representative papers",
    ]
    for paper in scholar.get("field_papers") or []:
        lines.append(
            f"- {paper.get('year', '')} · {paper.get('title')} · {paper.get('venue', '')} · citations {paper.get('citations', 0)}"
        )
    lines.extend(["", "## Summary", scholar.get("llm_blurb", "")])
    return "\n".join(lines)


def upload_dynamic_profile(
    client: Mixedbread,
    store_identifier: str,
    q: str,
    scholar: dict[str, Any],
) -> None:
    source = scholar.get("_source") or f"dynamic/{scholar['id']}.md"
    metadata = {
        "source": source,
        "kind": "scholar_profile",
        "stem": re.sub(r"[^a-z0-9]+", "_", scholar["id"].lower()).strip("_"),
        "scholar_id": scholar["id"],
        "dynamic": True,
        "query": q,
    }
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            suffix=".md",
            prefix=f"{metadata['stem']}_",
            encoding="utf-8",
            delete=False,
        ) as tmp:
            tmp.write(scholar_markdown(scholar))
            tmp_path = Path(tmp.name)
        client.stores.files.upload_and_poll(
            store_identifier=store_identifier,
            file=tmp_path,
            metadata=metadata,
            external_id=source,
            overwrite=True,
            poll_timeout_ms=60000,
        )
    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink()


def ensure_query_materialized(
    q: str,
    client: Mixedbread,
    store_identifier: str,
) -> dict[str, Any]:
    key = query_key(q)
    cache = load_dynamic_cache()
    if key in cache.get("queries", {}):
        return dynamic_payload_for_query(q)

    candidates, papers = discover_author_candidates(q, dynamic_scholar_limit())
    scholars = [materialize_scholar(q, candidate) for candidate in candidates]
    selected_ids = {scholar["id"] for scholar in scholars}
    edges = dynamic_edges_from_papers(selected_ids, papers)

    for scholar in scholars:
        try:
            upload_dynamic_profile(client, store_identifier, q, scholar)
            scholar["_upload_status"] = "completed"
        except Exception as exc:
            scholar["_upload_status"] = "failed"
            scholar["_upload_error"] = str(exc)

    cache = load_dynamic_cache()
    for scholar in scholars:
        cache["scholars"][scholar["id"]] = scholar
    existing_edge_keys = {
        (edge.get("source"), edge.get("target"), edge.get("type"))
        for edge in cache.get("edges", [])
    }
    for edge in edges:
        edge_key = (edge.get("source"), edge.get("target"), edge.get("type"))
        if edge_key not in existing_edge_keys:
            cache["edges"].append(edge)
    cache.setdefault("queries", {})[key] = [scholar["id"] for scholar in scholars]
    save_dynamic_cache(cache)
    return dynamic_payload_for_query(q)


def store_search(client: Mixedbread, store_identifier: str, q: str, top_k: int) -> list[Any]:
    response = client.stores.search(
        query=q.strip(),
        store_identifiers=[store_identifier],
        top_k=top_k,
        search_options={"rerank": True},
    )
    return object_field(response, "data", [])


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"ok": "true"}


@app.get("/api/rag-search")
def rag_search(
    q: str = Query(..., min_length=1),
    field: str = Query("learning-to-rank"),
    top_k: int = Query(20, ge=1, le=50),
    realtime: bool = Query(True),
) -> dict[str, Any]:
    if field != "learning-to-rank":
        raise HTTPException(status_code=404, detail=f"Unknown field: {field}")

    client = load_client()
    store_identifier = load_store_identifier()
    realtime_payload: dict[str, Any] = {"scholars": [], "edges": []}
    realtime_error = None
    if realtime:
        try:
            realtime_payload = ensure_query_materialized(q.strip(), client, store_identifier)
        except Exception as exc:
            realtime_error = str(exc)

    raw_items = store_search(client, store_identifier, q, top_k)
    results = []
    seen = set()
    realtime_ids = {scholar["id"] for scholar in realtime_payload.get("scholars", [])}
    for rank, item in enumerate(raw_items, start=1):
        result = normalize_result(rank, item)
        if not result or result["scholar_id"] in seen:
            continue
        if result.get("dynamic") and not (result["scholar_id"] in realtime_ids or scholar_by_id(result["scholar_id"])):
            continue
        seen.add(result["scholar_id"])
        results.append(result)

    for scholar in realtime_payload.get("scholars", []):
        if scholar["id"] in seen:
            continue
        seen.add(scholar["id"])
        results.append(
            {
                "scholar_id": scholar["id"],
                "rag_rank": len(results) + 1,
                "rag_score": 0.35,
                "source": scholar.get("_source"),
                "snippet": scholar.get("research_line", "")[:360],
                "dynamic": True,
            }
        )

    return {
        "query": q,
        "field": field,
        "results": results,
        "scholars": realtime_payload.get("scholars", []),
        "edges": realtime_payload.get("edges", []),
        "realtime_error": realtime_error,
    }


@app.get("/api/dynamic-data")
def dynamic_data() -> dict[str, Any]:
    cache = load_dynamic_cache()
    return {
        "scholars": list(cache.get("scholars", {}).values()),
        "edges": cache.get("edges", []),
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(ROOT / "index.html")


@app.get("/{page_name:path}")
def static_page(page_name: str) -> FileResponse:
    allowed = {"index.html", "list.html", "graph.html", "DEMO-GUIDE.md"}
    if page_name in allowed:
        return FileResponse(ROOT / page_name)
    raise HTTPException(status_code=404, detail="Not found")
