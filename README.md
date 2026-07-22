Improvements over time:

1) Added Langsmith tracing

2) Identified that translation task is consuming lot of time if video is 1hr long or above.

        Approach  was to solve it by chunking first and send raw non-translated chunks asynchronously to LLM. got an 8x improvement in latency.


Preparing the evaluation dataset : 

1) 