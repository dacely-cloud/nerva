use crate::json::json_escape;

#[derive(Clone, Debug, Eq, PartialEq)]
struct AcceptanceCheck {
    name: &'static str,
    passed: bool,
    details: String,
}

impl AcceptanceCheck {
    fn to_json(&self) -> String {
        format!(
            "{{\"name\":\"{}\",\"passed\":{},\"details\":\"{}\"}}",
            self.name,
            self.passed,
            json_escape(&self.details),
        )
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct AcceptanceReport {
    checks: Vec<AcceptanceCheck>,
}

impl AcceptanceReport {
    pub(crate) fn push(&mut self, name: &'static str, passed: bool, details: impl Into<String>) {
        self.checks.push(AcceptanceCheck {
            name,
            passed,
            details: details.into(),
        });
    }

    pub(crate) fn push_audit_result(&mut self, name: &'static str, result: (bool, String)) {
        let (passed, details) = result;
        self.push(name, passed, details);
    }

    pub(crate) fn passed(&self) -> bool {
        !self.checks.is_empty() && self.checks.iter().all(|check| check.passed)
    }

    fn passed_count(&self) -> usize {
        self.checks.iter().filter(|check| check.passed).count()
    }

    fn failed_count(&self) -> usize {
        self.checks.len() - self.passed_count()
    }

    pub(crate) fn to_json(&self) -> String {
        let mut items = String::from("[");
        for (index, check) in self.checks.iter().enumerate() {
            if index != 0 {
                items.push(',');
            }
            items.push_str(&check.to_json());
        }
        items.push(']');
        format!(
            "{{\"status\":\"{}\",\"acceptance_schema\":\"nerva-acceptance-v1\",\"checks\":{},\"passed\":{},\"failed\":{},\"items\":{}}}",
            if self.passed() { "ok" } else { "failed" },
            self.checks.len(),
            self.passed_count(),
            self.failed_count(),
            items,
        )
    }
}
