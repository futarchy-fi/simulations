# Proposal Poker — Formal Model Specification

## 1. Environment

### 1.1 Proposals

A proposal is a pair (x, y).

    x ~ N(0, 1)          quality (unobservable)
    y ~ LogNormal(0, 2)  importance (public to agents, not to mechanism)

x and y are independent. Each proposal is an independent one-shot game
with the same agent population and the same starting wealths.

### 1.2 Social objective

    Value(D) = sum_i  x_i * y_i * D_i

where D_i = 1 if proposal i is approved, 0 if rejected.

Oracle optimum:

    Value* = sum_i  x_i * y_i * 1(x_i > 0)

## 2. Agents

N agents indexed j = 1, ..., N. Each agent's type is a single
parameter: wealth W_j.

### 2.1 Wealth

    log(W_j) ~ N(mu_W, sigma_W)

Defaults: mu_W = 3.0, sigma_W = 1.5.

Wealth is fixed across proposals (each proposal is a fresh one-shot
game).

### 2.2 Signal precision

    tau_j = (phi / alpha) * W_j

Defaults: phi = 0.01, alpha = 0.005, giving tau_j = 2 * W_j.

### 2.3 Signals

For every proposal, every agent observes a private signal for free:

    s_j = x + epsilon_j,    epsilon_j ~ N(0, 1/tau_j)

Conditionally independent across agents given x. Agents observe s_j
and y before taking any action.

### 2.4 Participation cost

If an agent enters the betting pool for a proposal, they pay a
dead-weight monetary cost:

    K_j = phi * W_j * sqrt(y) * 1(participates)

Where `participates` means the agent has at least one accepted
contribution on that proposal. This money vanishes; it does not go
to the mechanism or to other agents.

### 2.5 Utility

Agent j's utility for a single proposal:

    U_j = log(W_j + T_j - K_j - fee_rate * S_j) - log(W_j)

Where:
- T_j = net monetary transfer (payouts received minus money contributed)
- K_j = participation cost paid on first accepted contribution
- S_j = total money contributed to the mechanism across all rounds
- fee_rate = 0.01 (1%) by default

Both `K_j` and `fee_rate * S_j` are dead-weight monetary costs. They
reduce terminal wealth directly and are not paid to the mechanism or
to other agents.

If agent j does not participate: T_j = 0, K_j = 0, S_j = 0,
U_j = 0.

### 2.6 Participation constraint

Agent j cannot participate in a proposal when:

    phi * sqrt(y) >= 1

This binds at y >= 10,000 for default phi = 0.01. It is the same for
all agents.

## 3. Verification Oracle (Futarchy)

For any proposal, a futarchy can be run at cost C. It produces:

    z = x + nu,    nu ~ N(0, 1/tau_F)

Defaults: C = 50, tau_F = 10.

The oracle is exogenous. If invoked, the futarchy signal z determines
the decision (approve if z > 0) and which side is "correct" for
settlement.

## 4. Mechanism Space

A mechanism defines the rules of a game. It has no access to the
environment — it cannot see x, y, s_j, W_j, tau_j, or N. It
observes only the contributions agents make.

### 4.1 Contributions

A contribution is a pair:

    (amount: float, data: Any)

where amount > 0 is money given to the mechanism, and data is a
payload whose schema is defined by the mechanism. The mechanism
specifies what data it accepts (e.g., binary YES/NO, a real number
in [0,1], a vector, or nothing at all).

### 4.2 Receipts

Each contribution produces a receipt:

    Receipt(id, amount, data, state_at_entry)

The receipt records the contribution's fields plus a snapshot of the
mechanism state at the time it was made. Receipts are the basis for
computing payouts. The mechanism never knows which agent holds which
receipt.

### 4.3 Mechanism definition

A mechanism M is a tuple of functions:

    init() -> State
        Initial state. Contains no proposal-specific information.

    publish(State) -> Message
        What to reveal to agents. Called after init and after each
        contribution. This is the only channel from mechanism to
        agents.

    on_contribution(State, Contribution) -> (State, Receipt)
        Process a single contribution. Update state, issue receipt.
        The mechanism may reject invalid contributions by returning
        the state unchanged and a null receipt.

    on_round_end(State) -> (State, bool)
        Called after each round (all agents have had a chance to
        act). The mechanism does not know how many agents acted or
        exist. Returns updated state and whether the game is done.

    outcome(State) -> (Decision, PayoutFn, bool)
        Called at termination.
        - Decision: "approve" or "reject"
        - PayoutFn: Receipt -> float (money returned to receipt holder)
        - bool: whether to invoke the futarchy oracle

    external_funding(State, Settlement) -> float
        Optional non-agent funding available at settlement. This can
        encode public subsidies or sponsor budgets. Defaults to 0.

    valid_data() -> Schema
        Defines what data the mechanism accepts in contributions.

If the futarchy oracle is invoked, its signal z overrides the
mechanism's decision (approve if z > 0) and determines which side
is correct for settlement. The PayoutFn may reference the oracle
outcome.

### 4.4 Mechanism constraints

1. **No environment access.** State, transitions, publications, and
   outcomes depend only on observed contributions. The mechanism
   cannot condition on x, y, s_j, W_j, tau_j, or N.

2. **Mechanism-defined funding.** Total payouts may exceed total
   contributions if the mechanism has public external funding defined
   by its own rules.

3. **Sybil resistance.** Two receipts with identical (amount, data,
   state_at_entry) receive identical payouts. The payout depends
   only on receipt fields, never on which agent holds the receipt.

4. **Voluntary participation.** Agents choose whether and when to
   contribute. The mechanism cannot compel contributions.

## 5. Simulation Protocol

For each proposal (x, y):

    1. Draw signals s_j for each agent j.
    2. state = mechanism.init()

    3. Repeat rounds until done:
       a. For each agent j in random order:
          - msg = mechanism.publish(state)
          - agent sees (W_j, s_j, y, public_history, own_past)
          - agent returns a Contribution or None
          - if Contribution:
              state, receipt = mechanism.on_contribution(state, c)
              assign receipt to agent j (invisible to mechanism)
              append msg to public_history for all agents
       b. state, done = mechanism.on_round_end(state)

    4. decision, payout_fn, use_futarchy = mechanism.outcome(state)

    5. If use_futarchy:
       - Draw z = x + N(0, 1/tau_F)
       - decision = "approve" if z > 0 else "reject"
       - Determine correct side from z for settlement

    6. For each agent j:
       - S_j = sum of amounts in j's contributions
       - payout_j = sum of payout_fn(r) for r in j's receipts
       - K_j = phi * W_j * sqrt(y) if j has any accepted contribution, else 0
       - T_j = payout_j - S_j
       - U_j = log(W_j + T_j - K_j - fee_rate * S_j) - log(W_j)

## 6. Agent Interface

An agent strategy is a function:

    agent(
        wealth: float,
        signal: float,
        y: float,
        public_history: list[Message],
        my_past: list[Contribution],
    ) -> Contribution | None

The agent knows its own wealth, signal, the public importance y,
the full history of mechanism publications, and its own past
contributions. It returns a contribution or None (do nothing).

The agent can derive its own precision from its wealth:
tau = (phi / alpha) * wealth.

## 7. Parameters

| Parameter       | Symbol  | Default | Description                        |
|-----------------|---------|---------|--------------------------------------|
| Quality         | x       | N(0,1)  | Proposal quality (unobservable)      |
| Importance      | y       | LN(0,2) | Proposal importance (public to agents)|
| Agents          | N       | 20      | Number of agents                     |
| Wealth          | W_j     | LN(3,1.5)| Agent wealth (fixed across proposals)|
| Entry cost frac | phi     | 0.01    | Participation constraint parameter   |
| Precision scale | alpha   | 0.005   | tau_j = phi * W_j / alpha            |
| Stake fee       | fee_rate| 0.01    | Dead-weight monetary fee per unit staked |
| Futarchy cost   | C       | 50      | Cost of verification oracle          |
| Futarchy prec   | tau_F   | 10      | Verification oracle precision        |
| Proposals       | M       | 500     | Number of proposals                  |

## 8. Assumptions

1. Agents act independently (no collusion).
2. Agents maximize expected utility (rational).
3. Signals are free. The cost is to act, not to observe.
4. Wealth is fixed across proposals (no accumulation).
5. The futarchy oracle is exogenous and independent.
6. y is public to agents, invisible to the mechanism.
7. Proposals are independent one-shot games.
8. The mechanism is anonymous — it cannot distinguish agents, only
   contributions.
