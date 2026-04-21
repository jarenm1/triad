#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StageKind {
    Intro,
    Straight,
    Offset,
    Turn90,
}

impl StageKind {
    pub const fn as_gpu(self) -> u32 {
        match self {
            Self::Intro => 0,
            Self::Straight => 1,
            Self::Offset => 2,
            Self::Turn90 => 3,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnDirection {
    Left,
    Right,
}

impl TurnDirection {
    pub const fn flag(self) -> u32 {
        match self {
            Self::Left => 1,
            Self::Right => 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct StageSpec {
    pub kind: StageKind,
    pub gate_count: u32,
    pub spacing: f32,
    pub lateral_amp: f32,
    pub turn_degrees: f32,
    pub radius: f32,
    pub vertical_amp: f32,
    pub hole_half_width: f32,
    pub hole_half_height: f32,
    pub direction: TurnDirection,
}

impl StageSpec {
    #[must_use]
    pub fn intro(gate_count: u32, spacing: f32) -> Self {
        Self {
            kind: StageKind::Intro,
            gate_count,
            spacing,
            lateral_amp: 0.0,
            turn_degrees: 0.0,
            radius: 0.0,
            vertical_amp: 0.0,
            hole_half_width: 0.14,
            hole_half_height: 0.14,
            direction: TurnDirection::Left,
        }
    }

    #[must_use]
    pub fn straight(gate_count: u32, spacing: f32) -> Self {
        Self {
            kind: StageKind::Straight,
            ..Self::intro(gate_count, spacing)
        }
    }

    #[must_use]
    pub fn offset(gate_count: u32, spacing: f32, lateral_amp: f32) -> Self {
        Self {
            kind: StageKind::Offset,
            lateral_amp,
            ..Self::straight(gate_count, spacing)
        }
    }

    #[must_use]
    pub fn turn90(gate_count: u32, radius: f32, direction: TurnDirection) -> Self {
        Self {
            kind: StageKind::Turn90,
            gate_count,
            spacing: 0.0,
            lateral_amp: 0.0,
            turn_degrees: 90.0,
            radius,
            vertical_amp: 0.0,
            hole_half_width: 0.14,
            hole_half_height: 0.14,
            direction,
        }
    }

    #[must_use]
    pub fn with_spacing(mut self, spacing: f32) -> Self {
        self.spacing = spacing;
        self
    }

    #[must_use]
    pub fn with_lateral_amp(mut self, lateral_amp: f32) -> Self {
        self.lateral_amp = lateral_amp;
        self
    }

    #[must_use]
    pub fn with_turn_degrees(mut self, turn_degrees: f32) -> Self {
        self.turn_degrees = turn_degrees;
        self
    }

    #[must_use]
    pub fn with_radius(mut self, radius: f32) -> Self {
        self.radius = radius;
        self
    }

    #[must_use]
    pub fn with_vertical_amp(mut self, vertical_amp: f32) -> Self {
        self.vertical_amp = vertical_amp;
        self
    }

    #[must_use]
    pub fn with_hole_half_extents(mut self, half_width: f32, half_height: f32) -> Self {
        self.hole_half_width = half_width;
        self.hole_half_height = half_height;
        self
    }

    #[must_use]
    pub fn with_direction(mut self, direction: TurnDirection) -> Self {
        self.direction = direction;
        self
    }
}

#[derive(Debug, Clone)]
pub struct CourseSpec {
    pub name: String,
    pub loop_enabled: bool,
    pub laps_required: u32,
    pub stages: Vec<StageSpec>,
}

impl CourseSpec {
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            loop_enabled: false,
            laps_required: 1,
            stages: Vec::new(),
        }
    }

    #[must_use]
    pub fn default_drone_course() -> Self {
        Self {
            name: "basic-lap".to_string(),
            loop_enabled: true,
            laps_required: 1,
            stages: vec![
                StageSpec::intro(4, 1.6),
                StageSpec::offset(5, 1.9, 0.75),
                StageSpec::turn90(3, 2.5, TurnDirection::Left),
                StageSpec::straight(4, 2.1),
                StageSpec::turn90(3, 2.5, TurnDirection::Left),
                StageSpec::offset(5, 1.9, 0.95),
                StageSpec::turn90(3, 2.5, TurnDirection::Left),
                StageSpec::straight(4, 2.1),
                StageSpec::turn90(3, 2.5, TurnDirection::Left),
            ],
        }
    }

    #[must_use]
    pub fn with_loop_enabled(mut self, loop_enabled: bool) -> Self {
        self.loop_enabled = loop_enabled;
        self
    }

    #[must_use]
    pub fn with_laps_required(mut self, laps_required: u32) -> Self {
        self.laps_required = laps_required.max(1);
        self
    }

    #[must_use]
    pub fn with_stage(mut self, stage: StageSpec) -> Self {
        self.stages.push(stage);
        self
    }

    pub fn push_stage(&mut self, stage: StageSpec) {
        self.stages.push(stage);
    }

    pub fn clear_stages(&mut self) {
        self.stages.clear();
    }

    #[must_use]
    pub fn total_gate_count(&self) -> u32 {
        self.stages.iter().map(|stage| stage.gate_count).sum()
    }

    #[must_use]
    pub fn compile(&self) -> CompiledCourse {
        let stages = self
            .stages
            .iter()
            .map(|stage| GpuStageSpec {
                kind: stage.kind.as_gpu(),
                gate_count: stage.gate_count,
                flags: stage.direction.flag(),
                _pad0: 0,
                spacing: stage.spacing,
                lateral_amp: stage.lateral_amp,
                turn_radians: stage.turn_degrees.to_radians(),
                radius: stage.radius,
                vertical_amp: stage.vertical_amp,
                hole_half_width: stage.hole_half_width,
                hole_half_height: stage.hole_half_height,
                _pad1: 0.0,
            })
            .collect::<Vec<_>>();

        let total_gate_count = self.total_gate_count();
        CompiledCourse {
            header: GpuCourseHeader {
                stage_count: stages.len() as u32,
                total_gate_count,
                loop_enabled: u32::from(self.loop_enabled),
                laps_required: self.laps_required.max(1),
            },
            stages,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuCourseHeader {
    pub stage_count: u32,
    pub total_gate_count: u32,
    pub loop_enabled: u32,
    pub laps_required: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuStageSpec {
    pub kind: u32,
    pub gate_count: u32,
    pub flags: u32,
    pub _pad0: u32,
    pub spacing: f32,
    pub lateral_amp: f32,
    pub turn_radians: f32,
    pub radius: f32,
    pub vertical_amp: f32,
    pub hole_half_width: f32,
    pub hole_half_height: f32,
    pub _pad1: f32,
}

#[derive(Debug, Clone)]
pub struct CompiledCourse {
    pub header: GpuCourseHeader,
    pub stages: Vec<GpuStageSpec>,
}
