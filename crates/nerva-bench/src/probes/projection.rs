use crate::parse::parse_optional_u32;

pub(crate) fn run_projection_bench(
    rows: u32,
    cols: u32,
    dtype: u32,
    iterations: u32,
    warmup_iterations: u32,
) -> Result<String, String> {
    let summary = nerva_cuda::projection::probe::projection_bench(
        dtype,
        rows,
        cols,
        iterations,
        warmup_iterations,
    );
    Ok(summary.to_json())
}

pub(crate) fn run_projection_bench_from_args(args: &[String]) -> Result<String, String> {
    let rows = parse_optional_u32(args.first().cloned(), 64, "rows")?;
    let cols = parse_optional_u32(args.get(1).cloned(), 128, "cols")?;
    let dtype = parse_optional_u32(args.get(2).cloned(), 1, "dtype")?;
    let iterations = parse_optional_u32(args.get(3).cloned(), 16, "iterations")?;
    let warmups = parse_optional_u32(args.get(4).cloned(), 2, "warmup_iterations")?;
    run_projection_bench(rows, cols, dtype, iterations, warmups)
}
