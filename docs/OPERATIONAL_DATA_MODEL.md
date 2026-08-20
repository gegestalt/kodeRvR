# Operational data model

The product's durable unit is an `OperationalRun`. It binds one analysis to an
exact `EvidenceTarget` and records the evidence needed to reproduce and review
the result.

```text
repository snapshot + change features
        |
        +--> evidence artifact IDs
        +--> model ID/version + feature schema + evaluation protocol
        +--> calibrated provenance estimate + uncertainty
        +--> public reuse matches and license metadata
        +--> decision outcome
        +--> optional human override with reviewer and reason
```

`OperationalRun` is persisted as versioned JSON by
`code_provenance.operational_data`. Evidence payloads remain in the
snapshot-bound Evidence Ledger; the run stores their stable artifact IDs. This
keeps the run compact while preserving auditability.

The record supports the product promise but does not turn an estimate into
authorship proof. `unknown`, OOD, abstained, and human-overridden outcomes stay
explicit and are never silently converted into labels.
