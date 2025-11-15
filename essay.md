# Part 1 — Theoretical Analysis

## Q1: Edge AI vs. Cloud AI — Latency and Privacy

Edge AI processes data and runs models on local devices (sensors, phones, vehicles) rather than sending raw data to remote cloud servers. This architecture yields two primary advantages:

- Latency reduction  
  In cloud-based systems, devices send data to a server, wait for processing, and receive a response — introducing network round-trip delays (hundreds of ms or more). Edge AI runs inference locally, removing transmission time and producing decisions in milliseconds, which is essential for real-time tasks.

- Privacy enhancement  
  Cloud systems transmit and store raw sensitive data (video, audio, personal information), increasing exposure to interception or server breaches. Edge AI keeps raw data on the device, sending only compact, anonymized results (e.g., “Person detected at 10:30 AM”), reducing the attack surface and preserving user privacy.

Real-world example — Autonomous drones:
- Latency: An onboard vision model detects and avoids obstacles in real time, enabling safe autonomous flight. Relying on cloud inference would introduce dangerous delays.
- Privacy: For inspections, an Edge-powered drone can analyze imagery locally and transmit only anonymized findings (e.g., “3 shingles missing at coordinates X,Y”) rather than full HD video of private property.

## Q2: Quantum AI vs. Classical AI in Optimization

Classical AI runs on classical computers using bits (0 or 1) and typically relies on sequential search or heuristics for combinatorial optimization. As problem size grows, possibilities explode and classical methods often settle for near-optimal solutions.

Quantum AI leverages qubits (superposition and entanglement) to explore many candidate solutions simultaneously. For certain optimization classes, quantum approaches (e.g., quantum annealing, variational algorithms) can find optimal or better solutions far faster than classical methods, especially as problem size scales.

Industries likely to benefit most:
- Pharmaceuticals & materials: faster molecular simulation and discovery of drugs or novel materials.
- Finance: improved risk modeling and portfolio optimization via advanced Monte Carlo-like evaluations.
- Logistics & supply chain: global route optimization, scheduling, and resource allocation at massive scale, reducing costs and emissions.