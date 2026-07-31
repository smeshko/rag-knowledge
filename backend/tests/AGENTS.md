# Testing guidelines

## Do not write tests for mocks, fakes, or stubs

Test doubles — mocks, fakes, stubs, and in-memory substitutes such as `FakeLLMProvider`, `FakeEmbeddingProvider`, and anything under `providers/<type>/fake.py` — are scaffolding to be *used in* other tests, not subjects to be tested themselves. Do not write a test whose subject under test is a fake. Asserting that a fake does what you just wrote it to do is tautological: it adds maintenance cost and protects no real behaviour.

If a fake is missing a capability a consumer needs, add the capability to the fake and exercise it through the consumer's test — never through a test of the fake itself.

## Contract suites are the exception — and they are not "testing the fake"

The abstract suites under `tests/contracts/` define the interface every implementation must satisfy. They bind no provider; a subclass supplies one. Running such a suite against a fake does exercise the fake, but the subject under test is the *contract*, not the fake — the identical suite runs against the real providers in later epics. Keep these suites, and keep binding fakes to them. Do not add fake-specific assertions into a contract suite, and do not copy a contract suite into a standalone fake test.
