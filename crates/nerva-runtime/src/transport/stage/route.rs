use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::id::transport::TransportDeviceId;

use crate::transport::stage::plan::StagePipelinePlan;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct StageEndpoint {
    pub stage_id: u32,
    pub transport_device: TransportDeviceId,
    pub lane_id: u32,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct StageBoundaryRoute {
    pub boundary_index: u32,
    pub source: StageEndpoint,
    pub destination: StageEndpoint,
    pub activation_bytes: usize,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct StageRouteValidationReport {
    pub route_validations: u64,
    pub route_identity_checks: u64,
    pub wrong_source_stage_rejections: u64,
    pub wrong_destination_stage_rejections: u64,
    pub non_adjacent_route_rejections: u64,
    pub endpoint_identity_rejections: u64,
    pub activation_size_rejections: u64,
}

impl StageRouteValidationReport {
    pub const fn route_rejections(self) -> u64 {
        self.wrong_source_stage_rejections
            + self.wrong_destination_stage_rejections
            + self.non_adjacent_route_rejections
            + self.endpoint_identity_rejections
            + self.activation_size_rejections
    }
}

pub fn planned_stage_routes(plan: &StagePipelinePlan) -> Vec<StageBoundaryRoute> {
    plan.boundaries
        .iter()
        .map(|boundary| StageBoundaryRoute {
            boundary_index: boundary.boundary_index,
            source: StageEndpoint {
                stage_id: boundary.source_stage,
                transport_device: TransportDeviceId(boundary.source_stage),
                lane_id: boundary.boundary_index,
            },
            destination: StageEndpoint {
                stage_id: boundary.destination_stage,
                transport_device: TransportDeviceId(boundary.destination_stage),
                lane_id: boundary.boundary_index,
            },
            activation_bytes: boundary.activation_bytes,
        })
        .collect()
}

pub fn validate_stage_route(plan: &StagePipelinePlan, route: StageBoundaryRoute) -> Result<()> {
    let boundary = plan
        .boundaries
        .iter()
        .find(|boundary| boundary.boundary_index == route.boundary_index)
        .ok_or_else(|| NervaError::InvalidArgument {
            reason: "stage route references an unknown boundary".to_string(),
        })?;

    if !plan
        .stages
        .iter()
        .any(|stage| stage.stage_index == route.source.stage_id)
    {
        return Err(NervaError::InvalidArgument {
            reason: "stage route source identity is not a known stage".to_string(),
        });
    }
    if !plan
        .stages
        .iter()
        .any(|stage| stage.stage_index == route.destination.stage_id)
    {
        return Err(NervaError::InvalidArgument {
            reason: "stage route destination identity is not a known stage".to_string(),
        });
    }
    if route.source.transport_device != TransportDeviceId(boundary.source_stage)
        || route.destination.transport_device != TransportDeviceId(boundary.destination_stage)
        || route.source.lane_id != boundary.boundary_index
        || route.destination.lane_id != boundary.boundary_index
    {
        return Err(NervaError::InvalidArgument {
            reason: "stage route endpoint identity does not match boundary".to_string(),
        });
    }
    if route.source.stage_id != boundary.source_stage {
        return Err(NervaError::InvalidArgument {
            reason: "stage route source identity does not match boundary".to_string(),
        });
    }
    if route.destination.stage_id != boundary.destination_stage {
        return Err(NervaError::InvalidArgument {
            reason: "stage route destination identity does not match boundary".to_string(),
        });
    }
    if route.destination.stage_id != route.source.stage_id.saturating_add(1) {
        return Err(NervaError::InvalidArgument {
            reason: "stage route must connect adjacent pipeline stages".to_string(),
        });
    }
    if route.activation_bytes == 0 || route.activation_bytes != boundary.activation_bytes {
        return Err(NervaError::InvalidArgument {
            reason: "stage route activation size does not match boundary".to_string(),
        });
    }
    Ok(())
}

pub fn probe_stage_route_validation(
    plan: &StagePipelinePlan,
) -> Result<StageRouteValidationReport> {
    let routes = planned_stage_routes(plan);
    for route in &routes {
        validate_stage_route(plan, *route)?;
    }

    let mut report = StageRouteValidationReport {
        route_validations: routes.len() as u64,
        route_identity_checks: routes.len().saturating_mul(2) as u64,
        wrong_source_stage_rejections: 0,
        wrong_destination_stage_rejections: 0,
        non_adjacent_route_rejections: 0,
        endpoint_identity_rejections: 0,
        activation_size_rejections: 0,
    };

    let Some(route) = routes.first().copied() else {
        return Ok(report);
    };

    let mut wrong_source = route;
    wrong_source.source.stage_id = wrong_source.source.stage_id.saturating_add(1);
    wrong_source.source.transport_device = TransportDeviceId(wrong_source.source.stage_id);
    report.wrong_source_stage_rejections =
        u64::from(validate_stage_route(plan, wrong_source).is_err());

    let mut wrong_destination = route;
    wrong_destination.destination.stage_id =
        wrong_destination.destination.stage_id.saturating_add(1);
    wrong_destination.destination.transport_device =
        TransportDeviceId(wrong_destination.destination.stage_id);
    report.wrong_destination_stage_rejections =
        u64::from(validate_stage_route(plan, wrong_destination).is_err());

    let mut non_adjacent = route;
    non_adjacent.destination.stage_id = non_adjacent.source.stage_id.saturating_add(2);
    non_adjacent.destination.transport_device =
        TransportDeviceId(non_adjacent.destination.stage_id);
    report.non_adjacent_route_rejections =
        u64::from(validate_stage_route(plan, non_adjacent).is_err());

    let mut wrong_endpoint = route;
    wrong_endpoint.source.transport_device = TransportDeviceId(
        wrong_endpoint
            .source
            .transport_device
            .0
            .saturating_add(1000),
    );
    report.endpoint_identity_rejections =
        u64::from(validate_stage_route(plan, wrong_endpoint).is_err());

    let mut wrong_size = route;
    wrong_size.activation_bytes = 0;
    report.activation_size_rejections = u64::from(validate_stage_route(plan, wrong_size).is_err());

    report.route_identity_checks = report
        .route_identity_checks
        .saturating_add(report.route_rejections());
    Ok(report)
}
