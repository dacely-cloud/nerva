use crate::perf::json::{optional_bool, required_f64, required_string};

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct PerfMeasurement {
    pub engine: String,
    pub workload: String,
    pub scope: String,
    pub measurement_id: String,
    pub tokens_per_second: f64,
    pub p99_ms: f64,
    pub comparable: bool,
}

impl PerfMeasurement {
    pub(crate) fn parse(label: &str, source: &str) -> Result<Self, String> {
        let schema = required_string(source, "schema")?;
        if schema != "nerva-perf-measurement-v1" {
            return Err(format!(
                "{label}: unsupported perf artifact schema {schema}"
            ));
        }
        let measurement = Self {
            engine: required_string(source, "engine")?,
            workload: required_string(source, "workload")?,
            scope: required_string(source, "scope")?,
            measurement_id: required_string(source, "measurement_id")?,
            tokens_per_second: required_f64(source, "tokens_per_second")?,
            p99_ms: required_f64(source, "p99_ms")?,
            comparable: optional_bool(source, "comparable", true)?,
        };
        measurement.validate(label)?;
        Ok(measurement)
    }

    pub(crate) fn matches_workload(&self, other: &Self) -> bool {
        self.workload == other.workload && self.scope == other.scope
    }

    pub(crate) fn beats(&self, baseline: &Self) -> bool {
        self.comparable
            && baseline.comparable
            && self.tokens_per_second > baseline.tokens_per_second
            && self.p99_ms < baseline.p99_ms
    }

    fn validate(&self, label: &str) -> Result<(), String> {
        if self.engine.is_empty() || self.workload.is_empty() || self.scope.is_empty() {
            return Err(format!("{label}: engine, workload, and scope are required"));
        }
        if self.measurement_id.is_empty() {
            return Err(format!("{label}: measurement_id is required"));
        }
        require_positive(label, "tokens_per_second", self.tokens_per_second)?;
        require_positive(label, "p99_ms", self.p99_ms)
    }
}

fn require_positive(label: &str, field: &str, value: f64) -> Result<(), String> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(format!("{label}: {field} must be a finite positive number"))
    }
}
