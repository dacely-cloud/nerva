#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum TokenPolicyPath {
    DeviceFastPath,
    HostPolicyPath,
    HybridValidationPath,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct TokenPolicyStep {
    pub token_index: u64,
    pub path: TokenPolicyPath,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TokenPolicyPlan {
    pub steps: Vec<TokenPolicyStep>,
}

impl TokenPolicyPlan {
    pub fn probe_plan() -> Self {
        Self {
            steps: vec![
                TokenPolicyStep {
                    token_index: 0,
                    path: TokenPolicyPath::DeviceFastPath,
                },
                TokenPolicyStep {
                    token_index: 1,
                    path: TokenPolicyPath::DeviceFastPath,
                },
                TokenPolicyStep {
                    token_index: 2,
                    path: TokenPolicyPath::HybridValidationPath,
                },
                TokenPolicyStep {
                    token_index: 3,
                    path: TokenPolicyPath::HostPolicyPath,
                },
                TokenPolicyStep {
                    token_index: 4,
                    path: TokenPolicyPath::DeviceFastPath,
                },
                TokenPolicyStep {
                    token_index: 5,
                    path: TokenPolicyPath::HybridValidationPath,
                },
            ],
        }
    }
}
