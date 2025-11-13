use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde_json::json;
use reqwest::Client;
use serde_json::Value;
use sqlx::{PgPool, Row};
use uuid::Uuid;

// cuvs stuff 
use cuvs::cagra::{Index, IndexParams, SearchParams};
use cuvs::raft::handle::DeviceResources;
use cuda::driver::Device;

// global state
static mut CUVS_INDEX: Option<Arc<RwLock<CagraIndex>>> = None;
static mut GPU_RES: Option<DeviceResources> = None;
static mut DB_POOL: Option<PgPool> = None;
static mut HTTP_CLIENT: Option<Client> = None;


async fn embed_grok(text: &str) -> Result<Vec<f32>, String> {
    let client = unsafe { HTTP_CLIENT.as_ref().unwrap() };
    let api_key = std::env::var("GROK_API_KEY").map_err(|_| "GROK_API_KEY not set".to_string())?;

    let body = json!({
        "input": text,
        "model": "grok-beta"  // 1024-dim
    });

    let resp: Value = client
        .post("https://api.x.ai/v1/embeddings")
        .header("Authorization", format!("Bearer {}", api_key))
        .json(&body)
        .send()
        .await
        .map_err(|e| e.to_string())?
        .json()
        .await
        .map_err(|e| e.to_string())?;

    let embedding = resp["data"][0]["embedding"]
        .as_array()
        .ok_or("Invalid embedding response")?
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();

    Ok(embedding)
}

#[pyfunction]
fn init_system(db_url: String, dim: usize) -> PyResult<()> {
  // init GPUs
  let device = Device::get(0).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
  let res = DeviceResources::new(&device).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
  unsafe { GPU_RES = Some(res); }

  // init cuVS cagra index
  let params = IndexParams::new(dim as u32, u32);
  let index = CagraIndex::new(params).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
  unsafe { CUVS_INDEX = Some(Arc::new(RwLock::new(index))); }

  // init db
  let pool = PgPool::connect(&database_url)
    .await
    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
  unsafe { DB_POOL = Some(pool); }

  // http client
  unsafe { HTTP_CLIENT = Some(Client::new()); }

  Ok(())
}

#[pyfunction]
async fn upsert(text: String, metadata: String) -> PyResult<u64> {
  let embedding = embed_grok(&text).await.map_err(|e| PyRuntimeError::new+err(e.to_string()))?;

  // insert into pgvector
  let db = unsafe { DB_POOL.as_ref().unwrap() };
  let row: (i64,) = sqlx::query_as(
    "INSERT INTO items (embedding, metadata) VALUES ($1, $2) RETURNING id"
  )
  .bind(&embedding)
  .bind(&meta_json)
  .fetch_one(db)
  .await
  .map)err(|e| PyRuntimeError::new_err(e.to_string()))?;

  let id = row.0 as u64;

  // add to cuvs gpu idx
  let index = unsafe { CUVS_INDEX.as_ref().unwrap().clone() };
  let mut index_guard = index.write().await;
  let gpu_res = unsafe { GPU_RES.as_ref().unwrap() };
  index_guard
    .insert(gpu_res, &embedding, id as u32)
    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

  Ok(id)
}

#[pyfunction]
async fn batch_upsert(texts: Vec<String>, metadatas: Vec<String>) -> PyResult<Vec<u64>> {
  if texts.len() != metadatas.len() {
    return Err(PyRuntimeError::new_err("texts and metadatas must match length"));
  }

  let embeddings = futures::future::join_all(
    texts.iter().map(|t| embed_grok(t))
  )
  .await
  .into_iter()
  .collect::<Result<Vec<_>, _>>()
  .map_err(|e| PyRuntimeError::new_err(e))?;

  let db = unsafe { DB_POOL.as_ref().unwrap() };
  let mut tx = db.begin().await.map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

  let mut ids = Vec::new();
  for (emb, meta_str) in embeddings.iter().zip(metadatas.iter()) {
    let meta_json: Value = serde_json::from_str(meta_str).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let row: (i64,) = sqlx::query_as(
      "INSERT INTO items (embedding, metadata) VALUES ($1, $2) RETURNING id"
    )
    .bind(emb)
    .bind(&meta_json)
    .fetch_one(&mut tx)
    .await
    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
  ids.push(row.0 as u64);
  }

  tx.commit().await.map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

  let index = unsafe { CUVS_INDEX.as_ref().unwrap().clone() };
  let mut index_guard = index.write().await;
  let gpu_res = unsafe { GPU_RES.as_ref().unwrap() };
  for (id, emb) in ids.iter().zip(embeddings.iter()) {
    index_guard.insert(gpu_res, emb, *id as u32).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
  }

  Ok(ids)
}

// hybrid search
#[pyfunction]
async fn search(
  query: String,
  top_k: usize,
  filter: Option<String>.
) -> PyResult<Vec<(u64, f32, String)>> {
  let embedding = embed_grok(&query).await.map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

  //cuvs gpu ann (overfetch)
  let index = unsafe { CUVS_INDEX.as_ref().unwrap().clone() };
  let index_guard = index.read().await;
  let gpu_res = unsafe { GPU_RES.as_ref().unwrap() };

  let search_params = SearchParams::new((top_k * 10) as u32, 64); //k, ef_search
  let mut neighbor_ids = vec![0u32; search_params.k as usize];
  let mut distances = vec![0f32; search_params.k as usize];

  index_guard
    .search(
      gpu_res,
      &embedding,
      &mut neighbor_ids,
      &mut distances,
      &search_params,
    )
    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

  let candidates: Vec<i64> = neighbor_ids.iter().map(|&id| id as i64).collect();

  // backup, pgvector filter + exact cosine rerank
  let db = unsafe { DB_POOL.as_ref().unwrap() };
  let mut qb = sqlx::query_builder::QueryBuilder::new(
    "SELECT id, embedding <=> $1 AS dist, metadata FROM items WHERE id = ANY($2)"
  );
  qb.push_bind(&embedding).push_bind(&candidates);

  if let Some(f) = filter {
    let filter_json: Value = serde_json::from_str(&f).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    qb.push(" AND metadata @> ").push_bind(&filter_json);
  }

  qb.push(" ORDER BY dist LIMIT ").push_bind(top_k as i32);

  let rows = qb
    .build()
    .fetch_all(db)
    .await
    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

  let results = rows
    .into_iter()
    .map(|row| {
        let id: i64 = row.get("id");
        let dist: f32 = row.get("dist");
        let meta: Value = row.get("metadata");
        (id as u64, dist, meta.to_string())
    })
    .collect();

    Ok(results)
}

//python module
#[pymodule]
fn vector_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(init_system, m)?)?;
    m.add_function(wrap_pyfunction!(upsert, m)?)?;
    m.add_function(wrap_pyfunction!(batch_upsert, m)?)?;
    m.add_function(wrap_pyfunction!(search, m)?)?;
    Ok(())
}
