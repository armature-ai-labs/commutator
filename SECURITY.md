# Security and safety reports

Commutator decides which model controls a robot, so some bugs can cause physical harm. Report these privately, not as public issues:

- a stop command or e-stop flag that doesn't route to `safety-controller`
- a guardrail that is dropped or weakened
- a route to a model that should have been rejected (offline robot sent to the cloud, unsafe model during elevated safety)
- anything that lets a request bypass the router

Use GitHub's private vulnerability reporting ("Report a vulnerability" under the Security tab). We aim to respond within 7 days.

Commutator is not a certified safety system. Every robot it controls needs an independent hardware emergency stop.
