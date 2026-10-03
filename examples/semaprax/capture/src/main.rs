// Copyright (c) Microsoft. All rights reserved.

use std::cell::RefCell;
use std::error::Error;
use std::fmt::Write;
use std::rc::Rc;

use semaprax::agent_runtime::{
    Agent, AgentBoundaryProbe, AgentCancellation, AgentHost, AgentProviderAttempt,
    AgentProviderDisposition, AgentProviderSink, AgentProviderUsage, AgentRunStatus,
    AgentToolResultSink,
};
use serde::Serialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

const REVISION: &str = "eec951eb1cce83e5e0f42edf97cbb5b8f3cffa2c";
const ACTION: &str = "{\"schema\":\"semaprax.agent-runtime-action.v1\",\"kind\":\"tool\",\"tool_id\":\"fixture.read\",\"arguments\":{\"query\":\"alpha\"}}\n";
const FINAL: &str = "{\"schema\":\"semaprax.agent-runtime-action.v1\",\"kind\":\"final\",\"message\":\"done\"}\n";
const ARGUMENTS: &str = "{\"query\":\"alpha\"}";
const TOOL_ID: &str = "fixture.read";
const RESULT: &str = "{\"value\":\"alpha\"}";

const PROFILE_DOMAIN: &[u8] = b"semaprax.agent-runtime.profile-digest.v1\0";
const TASK_DOMAIN: &[u8] = b"semaprax.agent-runtime.task-digest.v1\0";
const ACTION_DOMAIN: &[u8] = b"semaprax.agent-runtime.action-digest.v1\0";
const PROVIDER_RESPONSE_DOMAIN: &[u8] = b"semaprax.agent-runtime.provider-response-digest.v1\0";
const CALL_ID_DOMAIN: &[u8] = b"semaprax.agent-runtime.call-id.v1\0";

#[derive(Clone)]
struct Probe;

impl AgentBoundaryProbe for Probe {
    fn policy_epoch(&self) -> u64 {
        7
    }

    fn elapsed_ms(&self) -> u64 {
        0
    }
}

#[derive(Clone, Debug, Serialize)]
struct Dispatch {
    observed: bool,
    provenance: String,
    call_id: String,
    tool_id: String,
    arguments_json: String,
    result_json: Option<String>,
    remaining_deadline_ms: Option<u64>,
}

struct Host {
    attempt: usize,
    dispatches: Rc<RefCell<Vec<Dispatch>>>,
}

impl Host {
    fn new(dispatches: Rc<RefCell<Vec<Dispatch>>>) -> Self {
        Self {
            attempt: 0,
            dispatches,
        }
    }

    fn dispatch(
        &mut self,
        call_id: &str,
        tool_id: &str,
        arguments_json: &str,
        remaining_deadline_ms: Option<u64>,
        sink: &mut AgentToolResultSink,
    ) -> bool {
        self.dispatches.borrow_mut().push(Dispatch {
            observed: true,
            provenance: "semaprax_authorized_host_call".to_owned(),
            call_id: call_id.to_owned(),
            tool_id: tool_id.to_owned(),
            arguments_json: arguments_json.to_owned(),
            result_json: None,
            remaining_deadline_ms,
        });
        let Some(result) = execute_fixture(tool_id, arguments_json) else {
            return false;
        };
        if !sink.push(result.as_bytes()) {
            return false;
        }
        self.dispatches.borrow_mut()[0].result_json = Some(result);
        true
    }
}

impl AgentHost for Host {
    fn policy_epoch(&self) -> u64 {
        7
    }

    fn elapsed_ms(&self) -> u64 {
        0
    }

    fn boundary_probe(&self) -> Box<dyn AgentBoundaryProbe> {
        Box::new(Probe)
    }

    fn tokenize(&mut self, _: &str, request: &str) -> Option<u64> {
        Some(request.len() as u64)
    }

    fn attempt_provider(
        &mut self,
        _: &str,
        _: &str,
        request: &str,
        _: u64,
        sink: &mut AgentProviderSink,
    ) -> AgentProviderAttempt {
        let response = if self.attempt == 0 { ACTION } else { FINAL };
        self.attempt += 1;
        assert!(sink.push(response.as_bytes()));
        AgentProviderAttempt::new(
            AgentProviderDisposition::Succeeded,
            AgentProviderUsage::new(request.len() as u64, response.len() as u64, 0),
        )
    }

    fn invoke_tool(
        &mut self,
        call_id: &str,
        tool_id: &str,
        arguments_json: &str,
        sink: &mut AgentToolResultSink,
    ) -> bool {
        self.dispatch(call_id, tool_id, arguments_json, None, sink)
    }

    fn invoke_tool_with_deadline(
        &mut self,
        call_id: &str,
        tool_id: &str,
        arguments_json: &str,
        remaining_deadline_ms: u64,
        sink: &mut AgentToolResultSink,
    ) -> bool {
        self.dispatch(
            call_id,
            tool_id,
            arguments_json,
            Some(remaining_deadline_ms),
            sink,
        )
    }
}

fn execute_fixture(tool_id: &str, arguments_json: &str) -> Option<String> {
    (tool_id == TOOL_ID && arguments_json == ARGUMENTS).then(|| RESULT.to_owned())
}

fn domain_digest(domain: &[u8], bytes: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(domain);
    digest.update(bytes);
    finish_digest(digest)
}

fn stable_action_id(run_id: &str, turn: u64, tool_id: &str, arguments_json: &str) -> String {
    let mut digest = Sha256::new();
    digest.update(CALL_ID_DOMAIN);
    digest.update(run_id.as_bytes());
    digest.update(turn.to_be_bytes());
    digest.update(tool_id.as_bytes());
    digest.update(arguments_json.as_bytes());
    finish_digest(digest)
}

fn finish_digest(digest: Sha256) -> String {
    let mut output = String::from("sha256:");
    for byte in digest.finalize() {
        write!(&mut output, "{byte:02x}").expect("String writes are infallible");
    }
    output
}

fn status_text(status: AgentRunStatus) -> &'static str {
    match status {
        AgentRunStatus::Completed => "completed",
        AgentRunStatus::Cancelled => "cancelled",
        AgentRunStatus::DeadlineExceeded => "deadline_exceeded",
        AgentRunStatus::BudgetExhausted => "budget_exhausted",
        AgentRunStatus::ProviderFailed => "provider_failed",
        AgentRunStatus::ToolFailed => "tool_failed",
        AgentRunStatus::PolicyRejected => "policy_rejected",
    }
}

fn run_case(
    case_id: &str,
    profile: &'static str,
    task: &'static str,
    expected_status: AgentRunStatus,
    inject_dispatch_after_rejection: bool,
) -> Result<Value, Box<dyn Error>> {
    let dispatches = Rc::new(RefCell::new(Vec::new()));
    let host = Host::new(dispatches.clone());
    let mut agent = Agent::new(profile, host, AgentCancellation::new())
        .map_err(|diagnostics| format!("profile rejected: {diagnostics:?}"))?;
    let run = agent
        .run(task)
        .map_err(|diagnostics| format!("run failed: {diagnostics:?}"))?;
    if run.status() != expected_status {
        return Err(format!("{case_id}: unexpected status {:?}", run.status()).into());
    }

    let trace_value: Value = serde_json::from_str(run.trace())?;
    let run_id = trace_value["run_id"]
        .as_str()
        .ok_or("trace run_id is missing")?;
    let response_digest = domain_digest(PROVIDER_RESPONSE_DOMAIN, ACTION.as_bytes());
    let matching_events = trace_value["events"]
        .as_array()
        .ok_or("trace events are missing")?
        .iter()
        .filter(|event| {
            event["kind"] == "provider_attempt_finished"
                && event["output_digest"].as_str() == Some(response_digest.as_str())
        })
        .collect::<Vec<_>>();
    if matching_events.len() != 1 {
        return Err(format!("{case_id}: proposal is not bound to exactly one provider event").into());
    }
    let proposal_turn = matching_events[0]["turn"]
        .as_u64()
        .ok_or("proposal event turn is missing")?;
    let action_id = stable_action_id(run_id, proposal_turn, TOOL_ID, ARGUMENTS);

    if inject_dispatch_after_rejection {
        if run.status() != AgentRunStatus::PolicyRejected {
            return Err("fault injection requires a rejected Semaprax run".into());
        }
        dispatches.borrow_mut().push(Dispatch {
            observed: true,
            provenance: "fault_injection_after_policy_rejection".to_owned(),
            call_id: action_id.clone(),
            tool_id: TOOL_ID.to_owned(),
            arguments_json: ARGUMENTS.to_owned(),
            result_json: None,
            remaining_deadline_ms: None,
        });
        if let Some(result) = execute_fixture(TOOL_ID, ARGUMENTS) {
            dispatches.borrow_mut()[0].result_json = Some(result);
        }
    }

    let captured = dispatches.borrow();
    if case_id == "compliant" {
        if captured.len() != 1 || captured[0].call_id != action_id {
            return Err("authorized host call_id does not match stable action id".into());
        }
    } else if inject_dispatch_after_rejection {
        if captured.len() != 1 || captured[0].provenance != "fault_injection_after_policy_rejection" {
            return Err("fault-injected dispatch was not captured".into());
        }
    } else if !captured.is_empty() {
        return Err("denied action reached the Semaprax host".into());
    }

    let dispatch = captured.first().map_or_else(
        || {
            json!({
                "observed": false,
                "provenance": "none",
                "call_id": Value::Null,
                "tool_id": Value::Null,
                "arguments_json": Value::Null,
                "result_json": Value::Null,
                "remaining_deadline_ms": Value::Null,
            })
        },
        |item| serde_json::to_value(item).expect("Dispatch serialization is infallible"),
    );

    Ok(json!({
        "schema": "agent-lightning.semaprax-policy-record.v1",
        "case_id": case_id,
        "metadata": {
            "semaprax_revision": REVISION,
            "collector": "examples/semaprax/capture",
            "source": "real Semaprax Agent Runtime with an in-memory fixture host",
            "fault_injection": inject_dispatch_after_rejection,
            "fault_injection_provenance": if inject_dispatch_after_rejection {
                "external dispatch after Semaprax policy rejection"
            } else {
                "none"
            },
        },
        "profile": {
            "document": profile,
            "digest": domain_digest(PROFILE_DOMAIN, profile.as_bytes()),
        },
        "task": {
            "document": task,
            "digest": domain_digest(TASK_DOMAIN, task.as_bytes()),
        },
        "proposal": {
            "schema": "agent-lightning.semaprax-single-tool-proposal.v1",
            "turn": proposal_turn,
            "action_document": ACTION,
            "provider_response_digest": response_digest,
            "action_digest": domain_digest(ACTION_DOMAIN, ACTION.as_bytes()),
            "stable_action_id": action_id,
            "tool_id": TOOL_ID,
            "arguments_json": ARGUMENTS,
            "claimed_policy_compliance": "The provider claims this action complies with policy.",
        },
        "decision": {
            "status": status_text(run.status()),
            "trace": run.trace(),
            "trace_digest": run.trace_digest(),
            "evidence": run.evidence(),
            "evidence_digest": run.evidence_digest(),
        },
        "dispatch": dispatch,
    }))
}

fn main() -> Result<(), Box<dyn Error>> {
    let records = vec![
        run_case(
            "compliant",
            include_str!("../data/profile-allowed.json"),
            include_str!("../data/task-compliant.json"),
            AgentRunStatus::Completed,
            false,
        )?,
        run_case(
            "denied_not_dispatched",
            include_str!("../data/profile-denied.json"),
            include_str!("../data/task-denied.json"),
            AgentRunStatus::PolicyRejected,
            false,
        )?,
        run_case(
            "denied_but_dispatched",
            include_str!("../data/profile-denied.json"),
            include_str!("../data/task-violation.json"),
            AgentRunStatus::PolicyRejected,
            true,
        )?,
    ];
    println!("{}", serde_json::to_string_pretty(&records)?);
    Ok(())
}
