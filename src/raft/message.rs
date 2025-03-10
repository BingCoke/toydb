use super::{Entry, Status};
use crate::error::Result;

use serde_derive::{Deserialize, Serialize};

/// A message address.
#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub enum Address {
    /// Broadcast to all peers.
    Peers,
    /// A remote peer.
    Peer(String),
    /// The local node.
    Local,
    /// A local client.
    Client,
}

/// A message passed between Raft nodes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Message {
    /// The current term of the sender.
    /// 在这里表明发送者的term
    pub term: u64,
    /// The sender address.
    pub from: Address,
    /// The recipient address.
    pub to: Address,
    /// The message event.
    pub event: Event,
}

/// An event contained within messages.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Event {
    /// Leaders send periodic heartbeats to its followers.
    /// 心跳的时候确定commit_index term
    Heartbeat {
        /// The index of the leader's last committed log entry.
        commit_index: u64,
        /// The term of the leader's last committed log entry.
        commit_term: u64,
    },
    /// Followers confirm loyalty to leader after heartbeats.
    /// followers 在心跳之后确认对 leader 的忠诚度
    ConfirmLeader {
        /// The commit_index of the original leader heartbeat, to confirm
        /// read requests.
        commit_index: u64,
        /// If false, the follower does not have the entry at commit_index
        /// and would like the leader to replicate it.
        has_committed: bool,
    },

    /// Candidates solicit votes from all peers.
    /// 候选者向所有人征集选票
    SolicitVote {
        // The index of the candidate's last stored log entry
        last_index: u64,
        // The term of the candidate's last stored log entry
        last_term: u64,
    },
    /// Followers may grant votes to candidates.
    /// 跟随者授予候选者投票
    GrantVote,
    /// Leaders replicate a set of log entries to followers.
    /// 领导者复制一系列entry给跟随者
    ReplicateEntries {
        /// The index of the log entry immediately preceding the submitted commands.
        base_index: u64,
        /// The term of the log entry immediately preceding the submitted commands.
        base_term: u64,
        /// Commands to replicate.
        entries: Vec<Entry>,
    },
    /// Followers may accept a set of log entries from a leader.
    AcceptEntries {
        /// The index of the last log entry.
        last_index: u64,
    },
    /// Followers may also reject a set of log entries from a leader.
    RejectEntries,
    /// A client request.
    ClientRequest {
        /// The request ID.
        id: Vec<u8>,
        /// The request.
        request: Request,
    },
    /// A client response.
    ClientResponse {
        /// The response ID.
        id: Vec<u8>,
        /// The response.
        response: Result<Response>,
    },
}

/// A client request.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Request {
    Query(Vec<u8>),
    Mutate(Vec<u8>),
    Status,
}

/// A client response.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Response {
    State(Vec<u8>),
    Status(Status),
}
