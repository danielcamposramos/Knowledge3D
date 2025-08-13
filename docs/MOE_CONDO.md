# MOE Condo: Routing Queries Across Multiple Houses

A mixture-of-experts (MoE) "condo" lets the system map many houses to distinct experts and route queries to the appropriate house.

## Houses as Experts

- Each **house** hosts an expert focused on a specific knowledge domain.
- Requests carry features (topic, embedding, or metadata) that help a router choose the expert most likely to answer.

## Router and `condo.json`

- A top-level `condo.json` file links each house's URI to its expert information.
- Example structure:

```json
{
  "houses": [
    {"uri": "house://math", "expert": "algebra"},
    {"uri": "house://physics", "expert": "mechanics"}
  ]
}
```

- The router reads `condo.json` to dispatch queries to the house whose expert best matches the request.

## Scaling Considerations

- Shard houses or introduce hierarchical routers as the condo grows.
- Cache frequent routes and balance load across experts.
- Monitor usage to migrate or spin up new houses dynamically.

