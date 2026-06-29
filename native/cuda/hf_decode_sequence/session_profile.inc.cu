void stash_prefill_metrics(NervaCudaHfDecodeSequenceSession *session,
                           const NervaCudaHfDecodeSequenceResult *out) {
  session->pending_prefill_kernel_launches = out->kernel_launches;
  session->pending_prefill_device_elapsed_ns = out->device_elapsed_ns;
  session->pending_prefill_sync_calls = out->sync_calls;
  session->pending_prefill_graph_replays = out->graph_replays;
  session->pending_prefill_graph_launches = out->graph_launches;
  session->pending_prefill_graph_nodes = out->graph_nodes;
  session->pending_prefill_available = 1;
}

void drain_prefill_metrics(NervaCudaHfDecodeSequenceSession *session,
                           NervaCudaHfDecodeSequenceResult *out) {
  if (session->pending_prefill_available == 0) {
    return;
  }
  out->kernel_launches += session->pending_prefill_kernel_launches;
  out->device_elapsed_ns += session->pending_prefill_device_elapsed_ns;
  out->sync_calls += session->pending_prefill_sync_calls;
  out->graph_replays += session->pending_prefill_graph_replays;
  out->graph_launches += session->pending_prefill_graph_launches;
  if (out->graph_nodes == 0) {
    out->graph_nodes = session->pending_prefill_graph_nodes;
  }
  session->pending_prefill_available = 0;
  session->pending_prefill_kernel_launches = 0;
  session->pending_prefill_device_elapsed_ns = 0;
  session->pending_prefill_sync_calls = 0;
  session->pending_prefill_graph_replays = 0;
  session->pending_prefill_graph_launches = 0;
  session->pending_prefill_graph_nodes = 0;
}

cudaError_t profile_begin(NervaCudaHfDecodeSequenceSession *session) {
  return cudaEventRecord(session->profile_start, session->stream);
}

cudaError_t profile_end(NervaCudaHfDecodeSequenceSession *session,
                        uint64_t *bucket) {
  cudaError_t err = cudaEventRecord(session->profile_stop, session->stream);
  if (err != cudaSuccess) {
    return err;
  }
  err = cudaEventSynchronize(session->profile_stop);
  if (err != cudaSuccess) {
    return err;
  }
  float elapsed_ms = 0.0f;
  err = cudaEventElapsedTime(&elapsed_ms, session->profile_start,
                             session->profile_stop);
  if (err == cudaSuccess && elapsed_ms > 0.0f) {
    uint64_t elapsed_ns = static_cast<uint64_t>(elapsed_ms * 1000000.0f);
    *bucket += elapsed_ns == 0 ? 1 : elapsed_ns;
  }
  return err;
}
