# **Architectures of Agentic Deep Research: Systems Design, Model Context Protocol, and Skill Frameworks**

## **Abstract**

This paper presents a comprehensive systems-level analysis of agentic deep research platforms, exploring their architectural topologies, tool-integration protocols, and evaluation frameworks. With the transition from monolithic Retrieval-Augmented Generation (RAG) to long-horizon, autonomous multi-turn systems, AI research agents have shifted from passive query expansion to active, reflective, and collaborative knowledge synthesis. We examine the structural design of closed-source platforms including OpenAI Deep Research, Google Gemini Deep Research, and Perplexity Pro Deep Research, alongside open-source paradigms such as Stanford STORM, LangChain Open Deep Research, and RigorPilot. We investigate how the Model Context Protocol (MCP) standardizes data and tool access, and how dynamic skill frameworks govern runtime capabilities. Furthermore, we analyze advanced search topologies, including Plan-Space Monte Carlo Tree Search (Plan-MCTS) and Thompson Sampling for starting-point optimization. Finally, we review emerging evaluation frameworks such as DREAM and the behavioral data extracted from large-scale search logs to map the operational boundaries and technical mitigations of state-of-the-art research agents.

## **Keywords**

Agentic Workflows, Model Context Protocol, Plan-MCTS, Systematic Literature Reviews, LangGraph, DREAM Evaluation

## **Introduction**

The evolution of information retrieval systems has reached a critical inflection point. Traditional semantic search paradigms, which rely on converting a single user prompt into a static vector query and extracting localized text passages, are increasingly inadequate for solving open-ended, complex analytical tasks1. These legacy frameworks fail to capture domain-specific conventions, provide coarse context that pollutes downstream generation, and lack the capacity to execute non-linear, multi-step search strategies1.  
To overcome these structural limitations, the field of artificial intelligence has transitioned toward agentic deep research platforms. By mid-2026, the convergence of advanced, highly deliberative thinking models—such as the GPT-5.2 engine, the o3-deep-research series, and Google's Gemini deep-research-max-preview models—has decoupled raw token generation from strategic, multi-turn thought processes2. This shift is characterized by agent-native APIs, including OpenAI's Responses API and Google's Interactions API, which provide built-in state persistence, sandboxed computing runtimes, and standardization protocols4.  
These platforms do not simply summarize search results; they autonomously plan, draft, verify, and compile comprehensive, analyst-grade reports by executing recursive search and retrieval loops spanning minutes to hours2. Consequently, the primary engineering challenge has migrated from improving the baseline linguistic capabilities of language models to optimizing the systems-level orchestrations, protocol integrations, and verification frameworks that gain domain-specific expertise8.

## **A Taxonomy of Deep Research**

To contextualize the behavioral patterns of agentic deep research platforms, it is necessary to establish a clear taxonomy of information synthesis workflows. Table 1 contrasts the characteristics of purely autonomous AI-driven systems, traditional human-driven Systematic Literature Reviews (SLRs), and hybrid human-in-the-loop (HITL) workflows.

### **Comparative Taxonomy of Deep Research Workflows**

| Dimension | Purely Autonomous AI Deep Research | Traditional Human-Driven SLRs (PRISMA) | Hybrid Human-in-the-Loop Curation |
| :---- | :---- | :---- | :---- |
| **Orchestration Mechanism** | Recursive autonomous agentic loops and tool-calling graphs3. | Rigid, manual protocols governed by multi-investigator reviews12. | Bidirectional collaborative planning and real-time path steering4. |
| **Search Paradigm** | Multi-query expansion, semantic routing, and programmatic fetching1. | Exhaustive boolean and MeSH indexing across clinical databases12. | Directed exploratory querying with structural conceptual maps13. |
| **Traceability and Auditing** | Inline metadata generation, execution tracking, and parsed URL caching3. | Stringent registry documentation (e.g., PROSPERO) and study flow charts12. | Interactive history auditing, draft reviews, and selective path pruning4. |
| **Error and Bias Mitigation** | Adversarial self-correction, verification steps, and citation routing17. | Double-blind multi-reviewer screening and validated bias scales (e.g., RoB 2\)12. | Dual human-AI verification, criteria adjusting, and selective grounding5. |
| **Context Compilation** | Automated text generation, file synthesis, and chart compilation5. | Manual narrative synthesis, meta-analyses, and statistical pooling12. | Co-authored outlines, visual graphs, and incremental draft polishing4. |

### **Traditional Systematic Review Frameworks and PRISMA Alignment**

In the non-AI domain, evidence synthesis relies on established reporting guidelines, most notably the Preferred Reporting Items for Systematic Reviews and Meta-Analyses (PRISMA 2020\) statement12. PRISMA establishes a multi-step workflow designed to minimize bias and ensure absolute reproducibility:

1. **Protocol Registration**: Prior to database traversal, the research objectives, eligibility criteria, and synthesis methods are logged in public registries such as PROSPERO12.  
2. **Systematic Database Search**: Exhaustive search strategies are executed across multiple indexes (e.g., PubMed, Embase, ClinicalTrials.gov), with precise records of terms, filters, and results12.  
3. **Multi-Stage Screening**: Records are screened sequentially—first by title and abstract, and second by full-text review—against predefined eligibility criteria, recording explicit exclusion reasons at each stage12.  
4. **Data Extraction & Quality Assessment**: Reviewers extract study characteristics and assess potential risk of bias using validated scales like RoB 2 or ROBINS-I12.  
5. **Synthesis**: Extracted findings are pooled quantitatively via meta-analysis or synthesized through a structured narrative approach12.

Modern research agents attempt to emulate and streamline these rigorous steps15. For example, Elicit Systematic Review aligns with PRISMA 2020 by automating high-sensitivity screening (97% sensitivity on abstracts, 99.5% on full text) while documenting explicit, auditable exclusion reasons and linking every extracted data point directly to verified coordinate locations in source documents15.  
Furthermore, the emergence of the PRISMA-trAIce (Transparent Reporting of Artificial Intelligence in Comprehensive Evidence Synthesis) checklist demonstrates the academic community's push to standardize the documentation of human-AI interaction, model parameters, and performance validations in automated evidence synthesis19.

### **Hybrid Human-in-the-Loop Curation**

To bridge the gap between human expertise and machine scale, hybrid workflows have emerged13. This is typified by Co-STORM, an extension of the Stanford STORM project, which establishes a collaborative discourse protocol13. In this paradigm, multiple simulated expert agents engage with a human user via a shared, dynamic mind map13. The user can actively steer the direction of inquiry by injecting prompts that alter the focus of the expert models, mitigating the risk of runaway contextual exploration13.  
Similarly, Google's Gemini Deep Research utilizes a collaborative planning workflow4. When initialized with collaborative parameters, the agent returns an explicit proposed research plan instead of executing immediate search queries4. The human operator reviews and refines this plan across multiple turns before granting authorization for full execution4.

## **Architectural Analysis of Current Engines & Protocols**

### **Closed-Source Commercial Architectures**

#### **OpenAI Deep Research**

The production-grade execution of OpenAI Deep Research utilizes the Responses API to coordinate long-horizon search and analysis tasks3. The system employs a multi-step pipeline:

* **Prompt Expansion**: An intermediate processing model (typically a specialized instance of the GPT-4.1 series) ingests the raw user query and generates an expanded, highly detailed system prompt containing explicit constraints, required source types, and target parameters3.  
* **Execution Engine**: The expanded prompt is passed to the core thinking models (o3-deep-research or o4-mini-deep-research)3. These models are optimized for multi-step background processing, utilizing an extended execution timeout (typically configured up to 3600 seconds) to execute complex, nested tool loops3.  
* **Tool Access and Sandbox Execution**: The model is equipped with a Code Interpreter running in a sandboxed container to process uploaded files, perform data transformations, and execute complex statistical analyses3.  
* **Retrieval Anchoring**: To connect with external environments, the model integrates with local or remote vector files (allowing a maximum of two vector stores per session) and specialized search/fetch Model Context Protocol (MCP) servers configured with non-interactive approval parameters (require\_approval: "never") to ensure uninterrupted execution flows3.

#### **Google Gemini Deep Research Agent**

The Gemini deep research infrastructure is deployed exclusively via the Interactions API (utilizing the deep-research-preview-04-2026 or the comprehensive deep-research-max-preview-04-2026 models)4. Key structural parameters include:

* **Token Scale**: The model supports an input context of up to 1,048,576 tokens and an output token limit of 65,536 tokens, enabling the ingestion of massive document corpora21.  
* **Interactive Tooling & Caching**: By default, the agent has access to Google Search grounding, URL content parsing, and dynamic Python code execution5. It leverages implicit context caching to preserve retrieval records across long-running turns21.  
* **Autonomous Visual Synthesizers**: A unique capability of the Gemini Deep Research engine is its ability to generate base64-encoded visual charts, graphs, and infographics natively, integrating quantitative diagrams directly alongside cited narrative blocks5.

#### **Perplexity Advanced Deep Research**

Perplexity's architecture operates on a "search as code generation" philosophy, bypassing the rigid limitations of monolithic retrieval pipelines1. Rather than forcing the agent to query a static search API, the platform exposes individual retrieval primitives (such as lexical matchers, semantic vector retrievers, and source-priority filters) via a developer SDK1. The model dynamically compiles these primitives into a customized, query-dependent execution pipeline1.  
This is paired with a dynamic progress visualizer that streams live operational tracking (e.g., specific files being read, current synthesis stages) and a persistent workspace canvas that permits real-time, collaborative editing of the generated report22.

#### **Phind**

Phind optimizes its core pipeline for technical and software engineering research23. The system targets programming queries by generating functional code alongside detailed structural explanations, directly querying live developer documentation, Stack Overflow threads, and GitHub repositories in real time to resolve complex code-level execution dependencies23.

#### **Consensus**

Consensus is architected as an evidence verification engine limited strictly to peer-reviewed scientific databases23. Rather than conducting unstructured web crawls, it maps user queries against Semantic Scholar metadata and generates a high-level consensus summary23. This summary is complemented by a "Consensus Meter," which aggregates and visualizes the percentage of studies that support, oppose, or remain neutral regarding the target research hypothesis24.

#### **Elicit**

Elicit is a highly structured, academic literature analysis engine that leverages Semantic Scholar’s database of over 200 million papers23. Elicit's extraction pipeline focuses on systematic reviews and data extraction at scale, isolating methodologies, patient cohorts, interventions, and outcomes directly from document bodies15. The tool organizes these dimensions into highly traceable, tabular formats, permitting researchers to compare multiple studies simultaneously without manual extraction15.

### **Open-Source Deep Research Frameworks**

#### **GPT-Researcher**

GPT-Researcher leverages a parallelized multi-agent architecture to execute comprehensive web research26. The orchestrator generates detailed search tasks, spawns parallel worker queries to query multiple search engines (such as Tavily, Bing, and DuckDuckGo), and executes automated deduplication filters on the retrieved articles26. These filtered segments are then dynamically merged into structured document outlines and compiled into a single unified report26.

#### **Stanford STORM**

STORM addresses the challenge of synthesizing long-form, cited articles from scratch by separating the workflow into two distinct sequential phases13:

* **Pre-Writing Phase**: The engine automatically discovers diverse analytical angles by surveying outlines of similar topics13. It then initiates a simulated conversation between an expert agent grounded in internet retrieval and a simulated Wikipedia writer13. This conversation encourages the model to generate highly targeted follow-up queries, expanding its understanding of the domain13.  
* **Writing Phase**: Utilizing the compiled references and a generated outline, the system writes the full-length article, resolving citation anchors and utilizing a polishing step to eliminate redundant passages13. STORM is built on DSPy and supports customizable local and cloud vector storage backends, such as Qdrant13.

#### **RigorPilot Skills**

RigorPilot is a highly specialized, research-first agentic workflow designed for deep learning codebase experimentation and reproduction28. It strictly rejects generic coding agent patterns and structures its operations around a highly disciplined dual-lane framework28:

* **Trusted Lane**: Operates under strict, non-destructive safety boundaries28. It uses specialized skills such as ai-research-reproduction (to parse README command sequences), analyze-project (to map entrypoints and parameters without modification), and safe-debug (to analyze tracebacks and propose minimal adjustments)28.  
* **Explore Lane**: Triggered only upon explicit user authorization28. It executes candidate exploration (via ai-research-explore) and isolated codebase modifications (via explore-code) on dedicated git branches to preserve strict scientific comparability and avoid silent regression bugs28.

#### **LangChain Open Deep Research**

The open\_deep\_research project is structured as a sophisticated multi-agent StateGraph implemented in LangGraph11. To avoid the cognitive degradation that occurs when a single model attempts to manage broad, multi-topic context windows, the architecture implements a modular three-phase workflow11:

* **Scoping Phase**: The clarify\_with\_user node parses the incoming request11. If ambiguities, undefined acronyms, or overly broad parameters are detected, the node outputs a structured JSON schema requiring clarifying questions and halts execution until user feedback is received11. Once cleared, the write\_research\_brief node compiles the chat history into a highly focused research brief11.  
* **Research Phase**: The brief is passed to a central Supervisor agent, which delegates sub-tasks to parallel Researcher sub-agents14. Each researcher is constrained to its specific sub-topic to prevent context bloat14.  
  The sub-agents execute dynamic tool-calling loops (via Tavily, DuckDuckGo, or MCP tools) and invoke a final summarization step to compress raw scraped content before returning the data to the supervisor29. The supervisor utilizes an internal reflection tool (think\_tool) to evaluate the aggregated findings, decide if knowledge gaps remain, and spawn secondary parallel sub-agents if necessary14.  
* **Write Phase**: Once the supervisor invokes research\_complete, the compressed findings are routed to the final\_report\_generation node, which compiles the report in a single-shot generation step14.

#### **Firecrawl Deep Research Workflows**

Firecrawl deep research workflows leverage specialized API crawling primitives optimized for dynamic scraping and markdown extraction13. The system circumvents traditional scraping bottlenecks by systematically parsing JavaScript-heavy target websites into clean, JSON-structured Markdown layouts13. It feeds this high-density structured state directly to deep reasoning models, bypassing the raw textual noise that often triggers long-context retrieval failures13.

### **Model Context Protocol (MCP) in Deep Research**

The Model Context Protocol (MCP) establishes a standardized client-host-server communication layer built on bidirectional JSON-RPC 2.0 transport over either standard input/output (stdio) for local tools or Server-Sent Events (SSE) for remote APIs32. Within this framework, deep research agents interact with three core primitives: resources, tools, and prompts33. In production deep research systems, specialized MCP servers are deployed to handle specific cognitive and retrieval tasks:

* **Web Search and Scraping MCP Servers**: Expose search primitives (e.g., Brave Search, Google Search) and headless browser execution engines (e.g., Puppeteer, Playwright MCP)5. These servers convert active browsing, element clicking, and page fetching into standard JSON-RPC schemas33.  
* **Memory and Knowledge Graph MCP Servers**: Expose graph-based state-management structures, enabling the agent to maintain conceptual relationships, cross-references, and key entity mappings across multi-hour execution cycles5.  
* **Sequential Thinking MCP Server**: To enforce strict logical discipline, agents leverage the @modelcontextprotocol/server-sequential-thinking server36. This server exposes a single tool (sequential\_thinking) that requires the agent to log its current thought process through distinct, validated stages: Problem Definition, Research, Analysis, Synthesis, and Conclusion35.  
  By utilizing input parameters such as isRevision, revisesThought, and branchFromThought, the thinking agent can actively critique its prior conclusions, map alternative reasoning branches, and adjust its execution trajectory dynamically36.  
* **Developer and Repository MCP Servers**: Connect the agent to local or remote code repositories (e.g., GitHub MCP) and relational databases (Postgres/SQLite MCP), permitting live schema inspection, query execution, and code review directly inside the research harness33.

### **Agent Skill Frameworks and Runtime Tool Injection**

To extend an agent's capabilities dynamically without rewriting its core instructions, developers utilize modular skill package managers, such as Vercel Labs' npx skills (skills.sh)37 and Skilldex40. A skill is defined as a self-contained directory containing a highly structured SKILL.md instruction file accompanied by YAML metadata, required inputs, and output validation checklists9.  
Using npx skills add \<owner\>/\<repo\>, a developer can dynamically inject skills into specific agent directories (e.g., .claude/skills/ or .agents/skills/)38. Supported agents (including Claude Code, Cursor, and Copilot) automatically discover these skills at runtime37.  
Skilldex expands this model by introducing a three-tier hierarchical scope (global, shared, and project-level scopes) and providing verified badges for officially curated tools alongside a community-driven database40. Dynamic runtimes like Composio manage these capabilities by abstracting the underlying authentication plumbing (OAuth refreshes, database storage) and handling rate-limiting and retry logic, shielding the language model from execution errors and context window saturation31.

### **Dimensional Deep Dive: Key Research Architectures**

#### **Query Expansion**

Current systems vary drastically in how they expand and reformulate user queries. Commercial reasoning systems utilize internalized, non-linear reasoning steps to generate highly specific query sequences3. Conversely, modular systems like LangChain's Open Deep Research utilize dedicated intermediate steps to rewrite raw prompts, employing structural Pydantic schemas to output multi-perspective queries before launching parallel retrieval actions11.  
The most flexible approach is demonstrated by Perplexity’s programmable search architecture, where the model dynamically adjusts query blending ratios (such as lexical versus semantic weighting) based on the specific domain of the user query1.

#### **Browser Control and Crawling**

The mechanism of web navigation splits between raw headless browser automation and structured scraping services. While Puppeteer and Playwright MCP servers permit direct DOM manipulation, scrolling, and interaction with JS-heavy single-page applications, they introduce substantial latency, high execution costs, and susceptibility to anti-bot gating31.  
To mitigate these issues, systems increasingly delegate extraction to high-performance parsing layers like Firecrawl or Jina Reader13. These services compress dynamic layouts into compact, optimized Markdown formats, stripping style elements and scripts to prioritize high-speed semantic extraction13.

#### **Fact Verification**

To minimize hallucination rates, modern platforms implement multi-stage verification loops17. In open-source systems, this is represented by LangChain's Open Deep Research, which employs sub-agents to summarize and clean search findings before they are ingested by the supervisor, eliminating raw text noise and formatting errors29.  
Academic literature tools like Elicit and Consensus enforce traceability by mapping extracted claims to exact coordinates in source files and showing visual evidence markers (such as the Consensus Meter) to prove the analytical weight of a claim15.

#### **Long-Term Memory**

Maintaining coherent states over long-running, multi-hour research loops is managed via distinct memory topologies:

* **Internalized Context**: Flagship reasoning models manage state natively within expanded context windows3.  
* **Scratchpads and Outline Buffers**: STORM generates dynamic outlines where each section maintains a local research buffer, ensuring that writing nodes are grounded strictly in localized context segments13.  
* **Graph Databases**: Memory and Knowledge Graph MCP servers maintain relationships, entities, and citation anchors, permitting non-linear traversal and logical mapping5.  
* **Episodic Caching**: Systems like Mango archive navigation attempts, reflections, and trajectory failures in semantic stores to prevent redundant loops and guide future explorations42.

## **Architectural Comparison Matrix**

The comparative specifications, backends, memory architectures, and operational constraints of the analyzed platforms are structured in Table 2\.

### **Architectural Comparison Matrix**

| Platform / Protocol | Open / Closed | LLM Backend | Search APIs / MCPs Used | Memory Architecture | Planning Strategy | Key Strength | Key Weakness |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| **OpenAI Deep Research** | Closed | o3-deep-research, o4-mini-deep-research, GPT-5.22 | Web search, Code Interpreter, remote search/fetch MCP3. | Internalized high-logic reasoning state3. | Multi-step prompt expansion & autonomous execution3. | Exceptional logical synthesis and sandbox data execution3. | Long execution latencies, low real-time steering3. |
| **Google Gemini Deep Research** | Closed | deep-research-preview-04-2026, deep-research-max-preview-04-2026 \[cite: 4, 5\] | Google Search, URL context, Code Execution, remote MCP5. | Asynchronous state tracking, implicit context caching5. | Bidirectional collaborative planning (collaborative\_planning=True)4. | Multi-million token input window, native data visualization5. | Minimal support for complex custom structured schemas7. |
| **Perplexity Pro Deep Research** | Closed | Proprietary models & custom orchestrators1. | Programmable SDK primitives, sandbox calculator, dynamic crawling1. | Thread-level persistent canvas memory22. | Dynamic SDK primitive execution with active user questioning1. | Fast retrieval times, interactive workspace canvas22. | Unclear transparency of internal routing rules1. |
| **Stanford STORM** | Open | Configurable (OpenAI, Claude, DeepSeek, vLLM)16. | Google/Bing Search, Qdrant / VectorRM13. | Outline-driven local section research buffers13. | Perspective-guided simulated QA conversation13. | Strong document outline coherence, wiki-like reports13. | High reliance on the quality of initial simulated QA turns13. |
| **LangChain Open Deep Research** | Open | Configurable via init\_chat\_model()11. | Tavily, DuckDuckGo, custom MCP tools26. | LangGraph StateGraph, context-compressed briefs11. | Hierarchical Supervisor-Worker parallel worker delegation14. | Complete modularity, isolates context windows to avoid drift11. | High setup complexity, substantial token consumption11. |
| **RigorPilot Skills** | Open | Claude 3.5 Sonnet / Multi-model CLI wrappers28. | Git codebase APIs, trace tools, project analyzers28. | Git-based workspace snapshot state (current\_research)28. | Lane-restricted execution protocols (Trusted vs Explore)28. | Highly auditable, reproducible codebase adjustments28. | Bound strictly to software projects28. |

## **Frontiers: Theoretical Research Architectures**

### **Plan-MCTS: Plan-Space Search in Web Navigation**

Traditional web-traversal agents utilize search algorithms that operate directly on the atomic action space (e.g., clicking specific DOM elements, typing character strings)44. However, this approach faces severe "sparse valid path" challenges: the combinatorial space of interactive webpage components contains thousands of dead ends, causing the agent to exhaust its query budget on unproductive exploration44. Additionally, the noisy textual context of raw HTML pages dilutes the model's perception of its actual task progress44.  
To resolve these limitations, the **Plan-MCTS** framework reformulates web exploration by shifting the Monte Carlo Tree Search algorithm to a semantic **Plan Space**44. Instead of planning at the level of raw clicks, the agent reasons over high-level natural language intents (subplans), decoupling strategic decision-making from concrete execution grounding44.  
Let ![][image1] represent the current environment state, and let ![][image2] represent a semantic subplan generated from the plan space ![][image3]. During the selection phase of tree search, the agent traverses the plan tree by selecting the subplan ![][image4] that maximizes the Upper Confidence Bound Applied to Trees (UCT)44:  
![][image5]  
where ![][image6] represents the accumulated evaluation score of the subplan, ![][image7] is the visit count for that subplan edge, ![][image8] is the total visit count of the parent state node, and ![][image9] is the exploration constant that balances exploit-versus-explore behaviors.  
Plan-MCTS enforces stability through two key mechanisms:

1. **Dual-Gating Evaluation**: Every traversed subplan must pass a dual gate45. The first gate validates *physical completion* (whether the low-level executor successfully executed the actions), and the second gate validates *strategic contribution* (whether the outcome actively advances the global research goal)45.  
2. **Structural Refinement**: If a subplan fails physical execution, the system performs an on-policy repair, dynamically generating a localized alternative pathway to bypass the web interface failure without collapsing the global search tree45.

By distilling verbose, atomic execution logs into an **Abstracted Semantic History**, Plan-MCTS maintains precise state awareness, yielding state-of-the-art efficiency and task success rates on complex benchmarks like WebArena44.

### **Mango: Thompson Sampling for URL Selection**

When conducting research across large-scale websites containing thousands of nested pages, standard agents typically begin traversal from the home URL and drill down42. This forces the agent to traverse massive, irrelevant sub-trees and spend valuable compute resources on generic intermediate pages42.  
The **Mango** (Multi-Agent Navigation with Global-view Optimization) framework solves this by constructing a global structural map of the target website via a lightweight, high-speed crawler and site-specific search indexing42. Based on this map, Mango identifies a set of highly relevant candidate URLs to act as dynamic starting points42. To allocate its execution budget efficiently across these starting points, Mango models the URL selection process as a **Multi-Armed Bandit** problem and employs **Thompson Sampling**42.  
Let ![][image10] represent a candidate starting URL. The probability of success ![][image11] for navigating that URL to locate target information is modeled using a prior Beta distribution42:  
![][image12]  
At each exploration turn, the model samples a success probability from the distribution of each candidate URL and selects the URL with the highest sample value for execution42. Upon observing the trajectory outcome (where success is modeled as a binary reward ![][image13]), Mango updates the posterior distribution42:  
![][image14]  
![][image15]  
This trajectory data and its accompanying critical reflection are archived in an **episodic memory** component, preventing the model from initiating redundant visits to unproductive web branches42. Experimental evaluations on benchmarks such as WebVoyager and WebWalkerQA show that Mango achieves dramatic success rate improvements while lowering overall action counts42.

### **Reinforced Web Feedback & Adversarial Verifier Loops**

Frontier architectures are increasingly exploring Reinforcement Learning from Web Feedback (RLWF) to train deep search models directly17. In this paradigm, policy models are fine-tuned using reward signals mapped from successful web navigation traces17. This is coupled with **Adversarial Debater-Verifier Loops**, where a secondary, adversarial "debater" model critiques the intermediate findings generated by the primary research agent, attempting to find logical holes or unverified claims17.  
This dialectical process forces the research agent to run targeted, deep search cycles to find empirical evidence supporting its claims before outputting its final synthesized report, driving source hallucinations toward zero8.

## **Empirical Benchmarks & Performance Evaluation**

Evaluating agentic deep research platforms requires dynamic, open-ended evaluation protocols that go beyond traditional static benchmarks18.

### **Table 3: Performance of Representative Research Configurations on Diverse Datasets**

| Dataset / Metric | Evaluation Objective | Target Constraints Tested | Observed Performance Markers |
| :---- | :---- | :---- | :---- |
| **GAIA** | Multimodal, long-horizon assistant tasks5. | Complex web navigation, dynamic tool integration5. | Highly sensitive to reasoning model selection (e.g., o3 vs Claude)17. |
| **GPQA** | Graduate-level domain-specific Q\&A8. | Open-book research, complex scientific validation8. | Verification models reduce error rates under open-book constraints8. |
| **SWE-bench** | Software engineering repository resolution. | Sandbox code execution, multi-file code modifications. | High performance under strict "trusted-lane" safety boundaries28. |
| **BrowseComp** | Multi-step search trajectory efficiency17. | Query limits (typically capped at 600 tool calls)17. | Marco DeepResearch matches/exceeds several 30B-scale models17. |
| **Deep Research Bench** | Comprehensive PhD-level research synthesis26. | Multi-topic coverage, report formatting accuracy26. | LangChain Open Deep Research achieves a verified RACE score of 0.434411. |

### **Empirical Performance Case Study: Elicit systematic literature review evaluation**

To understand the baseline capability of automated evidence synthesis, Elicit evaluated its systematic literature review pipeline against a ground-truth dataset derived from 1,000 open-access Cochrane reviews, containing 38,493 study records balanced across 12 unique MeSH (Medical Subject Headings) categories15. The evaluation was structured across distinct, isolated phases:

* **Search Recall**: Grounded in a semantic search index of over 138 million papers and utilizing *only* the review titles as queries, a single Elicit semantic search retrieved **95%** of the target papers that were ultimately selected by human reviewers for the final Cochrane reports15.  
* **Abstract Screening Sensitivity**: When tested on a balanced abstract dataset containing 931 positive and 5,162 negative examples across 108 reviews, Elicit’s abstract screening models achieved **97% sensitivity and 93% specificity**, matching or exceeding the average performance of human dual-reviewer screening (98% sensitivity and 69% specificity)15.  
* **Full-Text Screening**: On a dataset consisting of 1,271 criterion-level judgements across 377 papers and 74 reviews, Elicit’s models registered a full-text screening sensitivity of **99.5%** and a specificity of **70%**15.  
* **Data Extraction Precision**: When given the task of extracting key parameters (Methods, Participants, and Interventions) across a golden test set of 769 extraction variables, Elicit achieved **96% accuracy**15. Crucially, every extracted data point was linked directly to source text coordinates, permitting human verifiers to audit decisions with a single click15.

### **The DREAM Evaluation Framework**

To solve the limitations of static evaluation metrics, the **DREAM** (Deep Research Evaluation with Agentic Metrics) framework introduces a dynamic, two-step protocol18:

1. **Protocol Creation**: An independent evaluation agent is given the target research query18. It crawls up-to-date web sources to construct a custom, query-adaptive grading checklist18.  
2. **Protocol Execution**: The grading agent actively reviews the report against three main metrics:  
   * **Key-Information Coverage (KIC)**: The agent converts key retrieved facts into binary yes/no questions to verify the presence of critical dimensions18.  
   * **Reasoning Quality (RQ)**: Evaluates structural cohesion, logical arguments, and causal claims by formulating validation plans that cross-reference the report’s narrative against factual ground truth18.  
   * **Factuality**: Evaluates presentation quality, structural clarity, and citation health, ensuring that the report is highly professional and free from hallucinated references18.

## **Technical Bottlenecks and Mitigations**

### **Multi-Step Context Drift**

As research agents execute recursive search loops, the accumulation of raw, scraped HTML code, failed API search outputs, and tangential search hits pollutes the model's context window, degrading analytical performance and inducing context drift1.

#### **Mitigation Protocol**

Developers should implement **context engineering via summarization**14. Rather than passing all scraped text directly to a central orchestrator, the system should leverage a multi-agent StateGraph where sub-topic researchers are spawned in isolated context windows14. These sub-agents execute localized tool calls and invoke a final summarization step to compress raw pages into clean, highly dense markdown summaries29. The supervisor only ingests these pre-processed, compressed summaries, preserving its context window and protecting its core reasoning state14.

### **Paywall Navigation and Dynamic Web Objects**

Static programmatic web scrapers frequently fail when encountering paywalled academic databases or complex single-page applications that require dynamic user actions, resulting in missing information and hallucinated summaries23.

#### **Mitigation Protocol**

Agents should be equipped with specialized browser-testing and browser-interaction skills (such as a Puppeteer or Playwright MCP server)34. The skill playbook should instruct the agent to execute logical user paths: opening the URL, detecting overlay elements, accepting cookie constraints, and waiting for dynamic Javascript elements to render34.  
This raw state is then parsed using dedicated scraping APIs (e.g., Firecrawl or Jina Reader) to extract clean Markdown segments, allowing the agent to bypass anti-bot detections and ingest accurate page bodies13.

### **API Rate-Limiting and Cost Overruns**

Complex deep research loops can run dozens of parallel tool calls and process millions of tokens, rapidly saturating API rate limits and causing unexpected, ballooning costs3.

#### **Mitigation Protocol**

* **Managed MCP Gateways**: Implement an enterprise MCP Gateway proxy between the agent clients and upstream tools41. The gateway enforces centralized caching, caching standard search results and document reads to prevent redundant API hits, reducing token costs by up to 90%7.  
* **Parallel Execution Semaphores**: Control costs and rate-limits by wrapping async parallel executions (e.g., asyncio.gather()) in strict semaphore boundaries, capping concurrent worker sub-agents to a manageable ceiling7.  
* **Multi-Tier Model Routing**: Deploy a cost-efficient model (e.g., o4-mini-deep-research) to handle high-turn user clarification, brief generation, and summarization tasks, reserving expensive high-logic engines (o3-deep-research or deep-research-max-preview-04-2026) strictly for final synthesis and adversarial evaluation loops3.

### **Citation Drift and Search Engine Bias**

Traditional search indexing prioritizes high-traffic web pages, biasing the research agent toward popular consensus rather than rigorous empirical data15. Additionally, models compiling long reports frequently experience "citation drift," misattributing findings to incorrect or completely hallucinated URLs15.

#### **Mitigation Protocol**

1. **Targeted Database Constraints**: Enforce strict source-selection guidelines during the scoping phase, restricting the search tool to high-authority academic databases (e.g., Semantic Scholar, PubMed, ClinicalTrials.gov)3.  
2. **Double-Review Screening**: Implement a dual-review screening workflow where the AI acts as a second reviewer to double-check inclusion and exclusion decisions, recording explicit quotes to justify each action15.  
3. **Literal Quote Linking**: Require the writing model to output the exact, literal quote supporting every assertion, alongside the source document's canonical metadata and coordinate reference, verifying that citations are clicking directly to live, active URLs3.

## **Conclusion & Future Directions**

The evolution of AI deep research platforms from simple, single-turn search engines to state-aware, autonomous systems represents a paradigm shift in digital knowledge curation1. This architecture is built upon the division of labor between high-level strategic planners and low-level, executable tools14.  
The Model Context Protocol establishes a standardized communication framework, while modular agent skill package registries permit the dynamic injection of capabilities on the fly31. Incorporating theoretical search models such as Plan-MCTS and Thompson Sampling-based URL selection prevents agents from getting trapped in unproductive web exploration paths, allowing them to optimize their search budgets over complex website hierarchies42.  
As we look toward the future, these autonomous systems will increasingly integrate reinforcement learning from web feedback (RLWF) and adversarial verifier loops to refine their search trajectories17. The development of rigorous, agentic evaluation frameworks like DREAM will continue to push the boundaries of accuracy, driving source hallucinations toward zero8.  
By mitigating operational bottlenecks through context summarization, managed MCP gateways, and literal quote-linking, agentic deep research platforms will establish a baseline for highly reliable, analyst-grade synthesis across academic, enterprise, and scientific landscapes3.

#### **Works cited**

1. Rethinking Search as Code Generation \- Perplexity Research, [https://research.perplexity.ai/articles/rethinking-search-as-code-generation](https://research.perplexity.ai/articles/rethinking-search-as-code-generation)  
2. ChatGPT Deep Research \- Wikipedia, [https://en.wikipedia.org/wiki/ChatGPT\_Deep\_Research](https://en.wikipedia.org/wiki/ChatGPT_Deep_Research)  
3. Deep research | OpenAI API, [https://developers.openai.com/api/docs/guides/deep-research](https://developers.openai.com/api/docs/guides/deep-research)  
4. Gemini Deep Research Agent | Gemini API \- Google AI for Developers, [https://ai.google.dev/gemini-api/docs/interactions/deep-research](https://ai.google.dev/gemini-api/docs/interactions/deep-research)  
5. How to use Deep Research with the Gemini API \- Google AI Studio, [https://aistudio.google.com/learn/deep-research-developer-guide](https://aistudio.google.com/learn/deep-research-developer-guide)  
6. OpenAI for Developers in 2025, [https://developers.openai.com/blog/openai-for-developers-2025](https://developers.openai.com/blog/openai-for-developers-2025)  
7. Google Gemini Deep Research API: What Developers Need to Know \- MindStudio, [https://www.mindstudio.ai/blog/google-gemini-deep-research-api](https://www.mindstudio.ai/blog/google-gemini-deep-research-api)  
8. Building Frontier Deep Research Systems in 2026 | Tomoro.ai, [https://tomoro.ai/insights/building-frontier-deep-research-systems-in-2026](https://tomoro.ai/insights/building-frontier-deep-research-systems-in-2026)  
9. SKILL-SCHEMA.md \- pnp/copilot-prompts \- GitHub, [https://github.com/pnp/copilot-prompts/blob/main/SKILL-SCHEMA.md](https://github.com/pnp/copilot-prompts/blob/main/SKILL-SCHEMA.md)  
10. The 2025 AI Agent Index Documenting Technical and Safety Features of Deployed Agentic AI Systems \- arXiv, [https://arxiv.org/html/2602.17753v1](https://arxiv.org/html/2602.17753v1)  
11. Building Enterprise Deep Research Agents with LangChain's Open Deep Research | by Tuhin Sharma | Medium, [https://medium.com/@tuhinsharma121/building-enterprise-deep-research-agents-with-langchains-open-deep-research-63e7cdb80a58](https://medium.com/@tuhinsharma121/building-enterprise-deep-research-agents-with-langchains-open-deep-research-63e7cdb80a58)  
12. PRISMA Guidelines: Step-by-Step Workflow \+ Examples \- Paperguide, [https://paperguide.ai/blog/prisma-guidelines/](https://paperguide.ai/blog/prisma-guidelines/)  
13. GitHub \- stanford-oval/storm: An LLM-powered knowledge curation system that researches a topic and generates a full-length report with citations., [https://github.com/stanford-oval/storm](https://github.com/stanford-oval/storm)  
14. Design Principles of Deep Research: Lessons from LangChain's OpenDeepResearch, [https://pub.towardsai.net/design-principles-of-deep-research-lessons-from-langchains-opendeepresearch-5d6432773281](https://pub.towardsai.net/design-principles-of-deep-research-lessons-from-langchains-opendeepresearch-5d6432773281)  
15. Elicit Systematic Review: Now Built for PRISMA 2020, [https://elicit.com/blog/systematic-review-for-prisma-2020](https://elicit.com/blog/systematic-review-for-prisma-2020)  
16. Stanford STORM Explained: AI That Writes and Curates Smarter | by Tarun Reddi \- Medium, [https://medium.com/predict/stanford-storm-explained-ai-that-writes-and-curates-smarter-ff39c746e290](https://medium.com/predict/stanford-storm-explained-ai-that-writes-and-curates-smarter-ff39c746e290)  
17. Marco DeepResearch: Unlocking Efficient Deep Research Agents via Verification-Centric Design \- arXiv, [https://arxiv.org/html/2603.28376v1](https://arxiv.org/html/2603.28376v1)  
18. DREAM: Deep Research Evaluation with Agentic Metrics \- arXiv, [https://arxiv.org/html/2602.18940v1](https://arxiv.org/html/2602.18940v1)  
19. Transparent Reporting of AI in Systematic Literature Reviews: Development of the PRISMA-trAIce Checklist \- JMIR AI, [https://ai.jmir.org/2025/1/e80247](https://ai.jmir.org/2025/1/e80247)  
20. Transparent Reporting of AI in Systematic Literature Reviews: Development of the PRISMA-trAIce Checklist \- PMC, [https://pmc.ncbi.nlm.nih.gov/articles/PMC12694947/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12694947/)  
21. Gemini Deep Research Agent | Gemini Enterprise Agent Platform | Google Cloud Documentation, [https://docs.cloud.google.com/gemini-enterprise-agent-platform/agents/google/deep-research](https://docs.cloud.google.com/gemini-enterprise-agent-platform/agents/google/deep-research)  
22. What's New in Advanced Deep Research | Perplexity Help Center, [https://www.perplexity.ai/help-center/en/articles/13600190-what-s-new-in-advanced-deep-research.html](https://www.perplexity.ai/help-center/en/articles/13600190-what-s-new-in-advanced-deep-research.html)  
23. 10 Best Perplexity Alternatives in 2026 (For Research and Productivity) \- Simular, [https://www.simular.ai/alternatives/perplexity-alternatives](https://www.simular.ai/alternatives/perplexity-alternatives)  
24. Best AI tools for medical research 2026: Elicit, Consensus, Semantic Scholar, Perplexity, and scite \- iatroX, [https://www.iatrox.com/blog/best-ai-tools-medical-research-2026-elicit-consensus-semantic-scholar-perplexity](https://www.iatrox.com/blog/best-ai-tools-medical-research-2026-elicit-consensus-semantic-scholar-perplexity)  
25. 6 Best AI Tools for Deep Research in 2026 (Tried and Tested) \- Index.dev, [https://www.index.dev/blog/ai-tools-for-deep-research](https://www.index.dev/blog/ai-tools-for-deep-research)  
26. langchain-ai/open\_deep\_research \- GitHub, [https://github.com/langchain-ai/open\_deep\_research](https://github.com/langchain-ai/open_deep_research)  
27. storm/examples/storm\_examples/README.md at main · stanford-oval/storm \- GitHub, [https://github.com/stanford-oval/storm/blob/main/examples/storm\_examples/README.md](https://github.com/stanford-oval/storm/blob/main/examples/storm_examples/README.md)  
28. lllllllama/RigorPilot-Skills \- GitHub, [https://github.com/lllllllama/ai-paper-reproduction-skill](https://github.com/lllllllama/ai-paper-reproduction-skill)  
29. Open Deep Research \- LangChain, [https://www.langchain.com/blog/open-deep-research](https://www.langchain.com/blog/open-deep-research)  
30. langchain-ai/deep\_research\_from\_scratch \- GitHub, [https://github.com/langchain-ai/deep\_research\_from\_scratch](https://github.com/langchain-ai/deep_research_from_scratch)  
31. Tool Calling Explained: The Core of AI Agents (2026 Guide) \- Composio, [https://composio.dev/content/ai-agent-tool-calling-guide](https://composio.dev/content/ai-agent-tool-calling-guide)  
32. What is Model Context Protocol (MCP)? A guide | Google Cloud, [https://cloud.google.com/discover/what-is-model-context-protocol](https://cloud.google.com/discover/what-is-model-context-protocol)  
33. What is Model Context Protocol (MCP)? \- IBM, [https://www.ibm.com/think/topics/model-context-protocol](https://www.ibm.com/think/topics/model-context-protocol)  
34. Sequential Thinking MCP \- Model Context Protocol Integration for Cursor IDE | MCPCursor, [https://mcpcursor.com/server/sequential-thinking](https://mcpcursor.com/server/sequential-thinking)  
35. Sequential Thinking \- Awesome MCP Servers, [https://mcpservers.org/servers/arben-adm/mcp-sequential-thinking](https://mcpservers.org/servers/arben-adm/mcp-sequential-thinking)  
36. @modelcontextprotocol/server-sequential-thinking \- NPM, [https://www.npmjs.com/package/@modelcontextprotocol/server-sequential-thinking](https://www.npmjs.com/package/@modelcontextprotocol/server-sequential-thinking)  
37. Skills.sh Manager \- Open VSX Registry, [https://open-vsx.org/extension/1fc0nfig/skills-sh-manager](https://open-vsx.org/extension/1fc0nfig/skills-sh-manager)  
38. vercel-labs/skills: The open agent skills tool \- npx skills \- GitHub, [https://github.com/vercel-labs/skills](https://github.com/vercel-labs/skills)  
39. skills.sh: npm for Agent Skills \- DEV Community, [https://dev.to/stevengonsalvez/skillssh-npm-for-agent-skills-35jc](https://dev.to/stevengonsalvez/skillssh-npm-for-agent-skills-35jc)  
40. Skilldex: A Package Manager and Registry for Agent Skill Packages with Hierarchical Scope-Based Distribution \- arXiv, [https://arxiv.org/html/2604.16911v1](https://arxiv.org/html/2604.16911v1)  
41. MCP Gateways: A Developer's Guide to AI Agent Architecture in 2026 | Composio, [https://composio.dev/content/mcp-gateways-guide](https://composio.dev/content/mcp-gateways-guide)  
42. Mango: Multi-Agent Web Navigation via Global-View Optimization \- arXiv, [https://arxiv.org/pdf/2604.18779](https://arxiv.org/pdf/2604.18779)  
43. Mango: Multi-Agent Web Navigation via Global-View Optimization \- arXiv, [https://arxiv.org/html/2604.18779v1](https://arxiv.org/html/2604.18779v1)  
44. Plan-MCTS: Plan Exploration for Action Exploitation in Web Navigation \- arXiv, [https://arxiv.org/pdf/2602.14083](https://arxiv.org/pdf/2602.14083)  
45. Plan-MCTS: Plan Exploration for Action Exploitation in Web Navigation \- arXiv, [https://arxiv.org/html/2602.14083v1](https://arxiv.org/html/2602.14083v1)  
46. \[2602.14083\] Plan-MCTS: Plan Exploration for Action Exploitation in Web Navigation \- arXiv, [https://arxiv.org/abs/2602.14083](https://arxiv.org/abs/2602.14083)  
47. Top 10 Agent Skills Every Developer Should Install \- Composio, [https://composio.dev/content/top-agent-skills](https://composio.dev/content/top-agent-skills)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAaCAYAAACO5M0mAAAAm0lEQVR4XmNgGAW0AsxALAbEwkDMiCYHBzlA/BaIDwHxFSC+gCqNALeBWBXKlgHiI0hycCAJxL+AeBIQawMxNxCLoKiAAhYg/o+EPwKxJ4oKJFAIxJeA+C8DRPFVVGkGBjUGiJUwIATEhxkgHoMDkJXLGVAdbgbEz4E4GEkMDAyA+AYQzwLifUD8CIjDGXCEI0gQ5HO8AT0KKAMA0PYZLAbjc+QAAAAASUVORK5CYII=>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADQAAAAZCAYAAAB+Sg0DAAABzElEQVR4Xu2VvytFYRjHv0IRJRGJxY+SWRHJZLAwGcgfwGCyGJEMTJIMDAaDlExkYDD6B5SUkpRBWBiQH99vz7nu9brn3s49l9zb+dSn03mf0znv+7zP+xwgIiIiH+mla/SEXtEL7z7mIi38ejpHmKQf9JiWfw+hFBabccZ/BX28gw45BmUHNullN+Dht9is0Ux36S3dhpXGLDJfkMpMkx50A6QSFjuiZU7sixp8r8taWpJwn4oCek1HkL3afqU3tMkNkHbYgibcgFA9ansHYA+90Gkv1k/f6Lh3n4wKeugOhqQINpd5Z1xJXqXvsO8mpQu2oFHYSzZosRdTJp5gW+tHH91yB0NSD5vLPazLyUdYsvdoZ/zRn3TD6lQLcWu2B7b15wljLupGc7QujUFQku5omxsIwil+1uwY4t3EjylY9mKZTKYOeBBUascI2cHcmm2EHfRnWFn60YrUJRkUJVSJ1S5lTOwQqjHEUCPQ2BKsi/mhWLpngqCS13cb3EAQqmEvWUF8Yg90Eyn6fALqOPrnqGOGQSW2DpuLOlrG6BxILaYKmb+shS7QM9ikEv0z9PPcR8ia/U+o519617zgAPEf6rATi4iIiMhtPgH0JWN2HTZ/xQAAAABJRU5ErkJggg==>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAaCAYAAAC+aNwHAAAA20lEQVR4XmNgGAXoQBOIpwHxLCB+BMUboHwQngTEbkDMCtOADkSAOACIy4H4HxDXAXEIFCcD8X4g/g/Ep2EacAFPIH4AxNJo4oxAPIUBYggLmhwccAPxHgaIK9ABSNMaBogB4mhycKAExM+B2AZdggHiogcMEANArsEK/Biw2wDSUAaVu4UmhwJaGbD7sRKI/wLxMSCWQZODA1A0vmWAxABZIJoBYvsDNHGiwRwGiAGgkCYZCDJAEgjIgHQ0OaJAKQNE81MgVkeTwwv0gfgTA0QzMgaJj4JRQB8AABhXLSG7tX8wAAAAAElFTkSuQmCC>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAcCAYAAABYvS47AAAAy0lEQVR4Xu2Qrw4BUBSHr42NIQmCKZrGVJMIgmIawVPwBF5AlhRPYARNVMmi5AkEvnP/2HFtqs1827fd+7tn95wdY/58jTQm1D2HBXW3NPGEGyzjAo/emi5aYhLvOA4PHsm6cuj4QxavWFVFghROdVDBPeZVJmcp7KnMtHGuA+N+f+syw74OYIgr4+a3yGrW2AgBZHCLdZXZ+S448nfZ5wRvzwqPtJWhU1g0btF6+RZZy864oT8S2h7ihxjZnbQ9Yyt6e2GgLEVvv8kDyKEfWFhqzL4AAAAASUVORK5CYII=>

[image5]: <data:image/png;base64,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>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEMAAAAaCAYAAADsS+FMAAADfUlEQVR4Xu2YS6hOURTHl1De77x1IwaiKKWIMqAYKHkkZeoxMKJLJN2SgQF5lCQRJa9kgEgGYkAZmJCBFCaSUEqRPNavdVZ333X3ud/5SuebfL/6d++31rl777P2Xmvt74q0adNqRkZDDfRTjY7GVsKCdqhORkcNMHenamB0lDFAdUB1JtGFwjc/2NFB1YjCfzz4Jhf2lHWqZ6rx0VETBIK1VQoI0Vus+q76KxaI9YWvo/j9keq9amvxLH8DG1XPVb/Fdj5OOFP1RrUi2OuGNbAplXkpFozV0aHsUT1QDQ12gkIQNge7c0x1TzU4OmqmS/VCmjidvCzB4MVT5qg+iR31WIzOqZYGW8oH1YxobAFsBptyRzUo+LKQHgTjRGLj2F9UnVa9U01KfHBKrObkYNLcaYoMExu30iID/I2nLDBW+jnlkDSxOR4ML55ArhOMNdI7GGPFakIZPHs2GgNei9i1r6pdPd19skQste+qponNRSq8Fiv8kbWqP6rl0ZFjn1gwLolFlyhfFUuTBWIFlp/O9uT3HDwbUy6FYO5X9Rebj/EITBW4s9xWzRVbM4HcIDbOTcnXB4L3S7Ut2LPwEAM/FAvEDbGIw1TpjioTHpXyoulQiHPF2CFQdCJegsCMEpu3CpzYVWIp+Fk1O/H5CY8b4RualoFSWDiD0IY44unO89k7DYNel8YLbxSMRWJjuq5I87dU8v+xanhi43OuK3ow0jJQigeD2kDk05edUPjo1Y06iNMoGEAAvkh3QCod4QROatxpTko8LeDBoBw0xB9GT4KPwLDYV9J3B0khR3dGo1iH2iLW+4G02ySWhvHFGkGHoDCmsM7L0nuN/n4xfbJw5Gg9sVA6TLJXyltXZJx0F+MU0oOd8zT06zJtOE0Tz/3cWsDX67WLeXZL+U2TU/dNNS86clAXSBG+e8QXgFyF7gvGoPURlBRe/rDYRY4Xfqu6ppqYPAPs+A8pTzVShNP0VKyGUSuYLwdrOS/5i2MWbmncJ8ZER0Gudzfip1i65ODCVOWyRdfIQYpwcgguNY2OlNtEYEMIVFew1wqnqSsam4ANioUQaKmkFelWhZWqj2L3kpZBPnORmh4dFaFrxUIIXi849lW4Jfalsezk1Aa14L6UF7Yy2P1Z0agsE/u3gmtKD29vOlQLo7GVEBC+7NXNENWRaGzTpk2b/8E/jDSunqu+06wAAAAASUVORK5CYII=>

[image7]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAAAaCAYAAAAHfFpPAAADa0lEQVR4Xu2Yy6tNURzHf0KRd6REiTxLKK+IEQPyGGCg/AEeeRRFUroTUwOPiRQGHkkZeKSIiwFSSnkl6pLIQEpRyOP7ub+z3LXX2fucfW+cMzmf+tY9a62919q/58KsRYv/xcCKGk1faUg62AxuSWPTwQbAx5+SeqUTZZgpXZfeSM+lrVb9or3SSeloJJ6LmS3NS8YayXBpp1WfvS6jpbXSdum39EmaFs3zwgXSNumLtE9aLg2I1uABjNjtzf8xOHFuOliWVdJrcyO0Zac66SOtTwcrbDTfvNmckdqtB3VokHTJ3PNEAEbYkllhtlAalowBhvspLUknmsAI6Ym5IXBYaaZKV8wfOmZugHuZFWYbkt9AyB+X3kpjkrkU1o6UeqcTdeA58jv26tDKWAprKYYd5qldGkKbD4H50jfpV9d0p2HORb8DRMQD8/yPa0IMh9okfZTuSx+kZZkVxdDejph79bs0WbotPZRemjsuZbf0wzxiS4PXQ373l66aRwEHAKx5p/J3zHjpvXmHKIL6QIRMMS+Wp622wWKWmnsUb7dLT6XplTnO+dg8qmJWmp99dTJeyCzpmmXzm4OSAofNPYhV90fzAZ6lMzCfxyjzwkqNOGT+0Ry4X7yoBtQhPj7sE6cBUcmH0rZj6p2pCnL7YDpovvk788J4wfKLXEiXos2IIFKHgyLCeLN1v11yRp6PoeAxhsdjumWAUDTywmWc+QZnzcM/r6iU2Qyvc4cg/3kfBsNwZQmFNjUAOU5dSevAHOmr1T7TX7DiTat+SSB47rLlh20wADUkDyJrcfT7hOV7rRahtVFrYtosv92FGrAjGa+CdkRFJdcnWH5YskGtiop3KWh0gvSOwBzFNG6Pz6Rd1rXXDOmz5bfdQAh/7idAWnEVD79TqFV5kZEh5G7wcPByCusIXbxQRJsVb0gBxDgc+Ib5XT10FqCoYWRa7qtoPIB3z5vPE2Wk6wvprjQpWhfgfe3mhqdLNIRgzKKLElWcbpC2qxiihw9Noe50VMTfvIdLUBE4AWekt9j/Cpa+aG71nkKnybtL0HnwflENSuHDuXNQwBvKIivOyXoQJQekdemEeSUnPUtVdPHIPM2aQihOcY6XgVwenA6Kieb/TA+qd31eY835z5gMe6QV6WAD4Epe1IpbtGjRIsMfw22ua4/+ChEAAAAASUVORK5CYII=>

[image8]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACwAAAAaCAYAAADMp76xAAACoElEQVR4Xu2WzYuNYRjGL6FoSCLTZEpkQVkoJJNZsTD5WMgOax/5WCiS0mzmHzCzkjKzGCRlMUNEhA1ZWYiNBYkspJQF8nFd5z5Pnvfued73nJNzpM6vrpp5Pt73fu77eu73AF3+X+bV1SgLqNl+sJM8oJb5wRK2U3epfj9RxlrYpjfUS+ooNaOwAjhDTVDnI2lfzHpqoxtrhH3UMzRx0KXUHuo49Yv6RK2J5hX8AHWM+kKdhWWmJ1qj0urQ/qCNIEtchSWhKXZRr2FBDxenasyi9vrBOgdhFWqVTdRnP1jGfGoallllWEEfKawANlML3ZjQQX9QW/1EkxyABd4Qq6mbsCxegAX8uLDCHuiRBS5Sb1F9cbR2CTXTT9TZQJ3ygzlUar1Y6JRfqZ9/pmsHkc88yvhTmH9jT8co0EPUR+oJ9YEaKqww+qhrsHdVoqwGf86lbsGyHHqkLuaj+t8xK6j3sA6SQ/5WBVbBLuclpA+o/q13yJ6lrKPuoOhPPViWGINlSKUaieYD2qvOkSulsqaLLI+PwoKULebEiyK0VntKkTfP+UHYpXsHu4jXkb5UwT65gEPLUrWkb9Rh5NtfZcDaOEnt9hNkOewlV2Clki08VRkWyqp6uPyr5+mAqW6gWCoDXkzdh3WJFCEzN5AuYwhYdyCFKrcl+n8c9ryd0VhAHn5F9fqJgNrLQ5hXVyJdpsvUd1gPTqHs6QKpU/gerTld3rjdvaBOIv0uXeCUNWsE74UMhix6tE6lVCVyDMNaVqpKKrEOo8/uPeoE8r/O9AHa5gfbQTh87sOyCOZLdYcylN2sHf4m6ttTsPK3igKVXTrGIOw3SKvo4/LcD7YbeVNezXk0hT5Qt2HeTl3EtnOa2uEHS1DH2I9/FGyXLp3iN4BnecqN9WEKAAAAAElFTkSuQmCC>

[image9]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAaCAYAAAC+aNwHAAABBUlEQVR4Xu3TP0uCURTH8RMlFPmHcMgmnQTBrcHNlxAiDeLkKEhr4N7SC2gRVwlaJchFRLfegJODg0ODNtma38N9wMej3tzzBx/Qc65HvR5F/kWiuEHE1O3zrTxhhl8s8INHXCCN3ProZvTAM6aoIRHU4+ighwmugvpGUhiKe1cdZJPEp7j+VrriGm/i/35lzG1Roy/+Rt42THTAwBavxQ14xZnp2RRQtMWKuAF3tnFIzvGOEWKmd1B0UQYBfexLHQ1b1LTl7wG6Dx+4tQ2N3sFY3GXui27hi3guWX/CJao4DdUv0RS3ZN5kZb1lOqyFPr7wEDrnzQkyKOFe3B8m/GmOOWZnVjnCKe2Ge9NDAAAAAElFTkSuQmCC>

[image10]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAbCAYAAACqenW9AAAAvUlEQVR4Xu3PsQpBURzH8b9QBsWgpJRXIFEewiyZPYNsRrPUnQ1SUt7AZJDNgOIZjMTE99x7j849984s91ef4f8//87/HJE4/0oGWbsZlS1OOMBBH6nAhJ8mhkgigSne4m0KpIyzBNe3xRsOZSThg0FETwo44mb01DtXVs+NXjc3eg08MEEXJX1Qxx0zv1afUx9VF/SwFG+7mxx2WPuDHTzxQgtjPahTxRUbLFDDBXtUjLlv0shbddGo4/woH3oEICHIxjH1AAAAAElFTkSuQmCC>

[image11]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABIAAAAbCAYAAABxwd+fAAABT0lEQVR4Xu2UvytFYRjHH6Eo8rMsJCYyKn+AKLPlKn8Ag8nAavEPiAVlMJHNQDIoGxbFJFmU/AEmhc/XOYfnfeuec67Bwqc+dc/3fc9zz32e9x6zP0cj9sRhLdRhBR/xEKfD5XKoyAI+4ViaqeDA146STOIrzrvsBWfddSlu8AK7XKZCq+66EDVWN41G+TuuRVkuU3iGLS5Tz1Ro0WWF6FsfcNO5g2844fbl0orneGJhIY1fExz83ppPryVjHo7yOTzAhiivihqsRne4TDeryIzLCskKeXQw1eia0Ojv3XUf3lnSJ089HuMeruOuhWfuc8xaaE7dxlNsi/YspVm2XwPSoAL68QivcR87w2Ubwef0c9a/qgdVP7E9DlOWLZms6MZb++GbQYX0FHpPqXeXOIQrbk8pdNbUty1LenWFGzjuN5VFU9MTiSYL/5f//CYfg+05Lq2E0sEAAAAASUVORK5CYII=>

[image12]: <data:image/png;base64,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>

[image13]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFUAAAAaCAYAAADG+xDjAAAC8ElEQVR4Xu2YS+hNURTGP6HIM29hRhKhFJFMKBkwwIAyUKZGJFPm/MlAkZIMJFKKUIow8CgzmVBIJEkUA/L4vtbd/a/VOfvsc+6597p1fvV1u3vte/faa++99gNoaGjojJHUDGo2NcbZeoXankqN8IZBZDT1nfpMXaTW/WvuGWpbPsiXA842UGhWHKcOU2OdrZ3J1ALYAHTKRGqeL2yh/z9E/cEAz9iZ1Etqrje0mEVdoT5QV6lP1F5YuiiL0spO6jV1ztnakS9vYb4NJMpj6qQ+PZOo29QTanqrbBv1k9ofKiUyRP2CLW3NwlhQYz51ld3UO1iHT1MnqT3Udmr9cLVCYh24DgvgWld+AhaYKhxEF4Pqdzn/PcYU2JLMW7JliHVA5d+oFa48BKYKXQvqSlge0yxbCBv5+9Q9WHBjjKduoVpOyyLWAQU0FtQqR6+uBFV56hq1GfbnX2GJ/0br+6bhqpnsgB096iJ0IGtTKAqqBrgsKUENm2dyUBU06TL1m9rQKlfemh8q5aDG7lLLYA3mqWi2Cx2PdJT6AVspWfQrqGIczDf5KF+TVsUz6hU1x5XHUMA0gtqg3kR0KfwgwiOY0+eRn5v7GVShfUMnBvm6xtky0a6q2TrKGyJMgDW0yBsqojOo8riCm0W/NiqxFeabfExGf6wGynIUloPrYgnsaqhc79EZtT1FBc6gu0GVLw9hviWj3KjlP80bEtEoLvWFFYnttEoLz2EzJgR9FWz2KtcFtEcoUBcQX3nKi8dgdWMpKuZTLlpOZ5F+Ls1Cg3IKNppJSTyHog4sp15Qj6l9sFkt37WRBPTbB9Qd5OfZMEO9smZskU+Z6IzZ6cOEHj90e7oJuye3O/q0rV4RKR2Qr3q5Unt5DyFCMzUvqGVI8em/ps4OHPEFFanTp75QVwe0clb7worU5VNf2Ui9hz3S6PW/bK5X/cW+sAK6tOyivlAfnW0g0QuXrr+6XGxxtl6htuWDfNFDdkNDQ0NDj/kLQoC2CtaHMb0AAAAASUVORK5CYII=>

[image14]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAmwAAABFCAYAAAD3qbryAAADi0lEQVR4Xu3dT6hnYxgH8EcoihCFsbkjG1ESsVE2FqawUShZs7AjCzsLS6UZKxbDwgpFCLsbFuzJxmJIKbOYElL+Pk/vnH7nvvP3PffU3OHzqW/d+54zp/fc1bf3PedMBAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADwP3R75oJ+EACAvePdzGX9IADAuVLF5PrMJf2BXarrvpO5tT+wgosz1/aDKxotbBdmrok2r+n36WcAgMWqVDySOZL5PPNb5rr5Cbt0f+bNWL+4fJQ5ltnOfJO5d8fRdYwWtvcyX2Z+yTwa7W/6Q+ae2TkAAEOqRP2VeXg2Vs9s/R2taH2c2T87NqqK3/zaa6jr1ZznBfChzD/RyuehzGOzY7sxUthqZe3GzG2ZnzNbmaujzeupzWkAAGOqSFQpu7Qbr5LxfLTys/Sh+8czP2b2RdtqHcnpHI22sjZ3R+bXaGWp7mfJNmk/h8onmZu6sbr2yVYLt6JtJ9d9199vUqtrVSQBAIZdkfki82R/IFrhqOJzS39gwE/RrvP9gpxOXfPDbqxWtqocHo7lq1n9HCp/RtvSnI99Ha0YnsrB2FnYAAAWm1alTraiVYWjnhHbrVqdq3JV5XANtT1Zc3uwG697+C7Wf7FhZEu0TH/T7W4cAGCRG6I9FH9VN14lq0rRG934Vub9zNuZ1zJP7Dh6ai9knukHF6q5/REnPsRfK161TdqXz61o9zHNedRoYZu2Q1/sDwAALFXPrtVboW9lvo22GnZ35rnM75kPoq2OVVHajlZeqizVatbNcXbq39ZW5X39gYUeiFbOXo32RuZX0e7jrsyn0d4YvTPzcmxWuqY5jxotbNvRSnCVYQCA1dSD8rUy1X9/7crYPCxfBaReQij1zNtnmcuP/342qvSMFJ8zqXnVnGuOc/P7OBInznnUaGGr+YycDwCwmno268Dxn2t7sR6sr89Y7GX1aY2a80WxmfP+HWecWa3S9UUWAGBPqm3N+t8Kno32EdzXMy/NT9iDno4251diM+f+8yUAAP8ptdVXW35V3upFhfPh22LT9uQ0ZwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADjv/QtctYIv+W9ohAAAAABJRU5ErkJggg==>

[image15]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAmwAAABFCAYAAAD3qbryAAAFXUlEQVR4Xu3dT6htVR0H8BUlFCqmTwxReI+HCiIRISmCb6ZUmAkNbCCOGqhQk5CsnAgh4kCQMIgstCAirEAiEWlwXzUJR4LlIIKSKApSEI1KUNeXdTZvv3W39/y5//blfT7w4927zrn37HfOYH9Zv7XWLQUAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADObRcsaq7O6wcAgKMnN/RjtT7QPzBTH6314X7wEJ2udbwfnJGLan25HJ3PFwAYyY38S7U+WNrN/NFaL5/1jHn5Qq0rF19fXesfte498/CBy/v36zIdhK6q9VSt6/sHDsmpWq/WuqF/AACYrw/V+k45O2w8UOs/o+/n5JO1fjf6/vJaf63149HYps4v06FrmYTFhKCxj9f6Y603ar1b5hPY8nn/pNZWNw4AzNittZ7pxh6u9VY3NgcfKe1ac82Dk6XNsP1wNLaphKp116B9rNYrtZ7oxtNezmNDoJxLYIvP1Pp/PwgAzNO1tZ6rdfFo7Iu1/lvr86OxTVxR61f94C59u9Zd3dhvar1Q1g9aUzYJbGl3/q2cadH2Djqwpa09bC7I15eOvh9kFjEzkvmMAICZSyh7vLSw8dqi/rkY36Q1OJbF7d/qB3cps2vX1bqvtJbtn2r9vNYl4yftwiaB7cXS1q+lnTrlIANb1tI9W+vNWp+r9Zdaz5cWKHtpe9/cDwIA85Obe1qKYz8rrcWXdt6mfl/2Z1F7WrW9t0sLh3thk8CW1vHUdQ0OKrBlJi0zjfk8s27uz6Xt+H2ptDV0vVtKC20AwMxlxqqfGcpNPCFok9mX/K6v17qjtKCyTq1yPMdn+4HSwlBCZhbTryPBrL+GT5e2q7Mf71uKYwlsOwWf/PwqgS3ty8vK9teeqqn36kRpwTWPJaClfRz5/0x9lrmena4bAJiJqZmh3Ohzw7+9f2AFD5bWqvx7absm16nbys6yzq6fDYx/l7bjcd2Zsa+U7dfwr9Lah/34JxY/M+V/Zefgs2pgS1v6dNn+2n1l5mzZe5XPb9kaRIENAI6AtNCmQsTrpa17yo7MTX2j1nfLzjNT67qnHyhtnd07pe163AubtkS/3w+OrBrY9kpeZ6ss/38kkH+1HwQA5uVTZXurLAErgW1Yf5ZAlMXrOTIjoeSni/Fl8nt+VJbP8qxq2NXYtz1vrPW9ciYYnqj1y9JapE/WunsxvqpNAtuyluxBB7bsop2aOe3leVMtZgBgJobDU/MXDbLB4AelnWWWf8fy1w6GtllCwDrHdFxY6xf94IYSLLPe7g+lbZRISMt13zl6TkLdVmmBK89PSMqxJevYJLA9VFpbtn+tIailPTmu/Qxuufatsvy4juF5u5lFBQD2WXaAZkdhDIvvs+i9l+MhMhOXhewJa988++Gl8rc+90JadwmO44X5vYSU4frSPv1taaFxHZsEtptKW8c21bI9DKu85wmXCZkAwIxlBioty2WywD3h7mRps0U5CiJnfR20XGvfvu0lbA0tvrQos3ki6/TWcU2Z3oG5k8xSpQ2bdX9HRXaTTp3NBgDMSNqLywJQZN3YI6XNXOVsteGoiIOW9u37rREbpCWaQ3TvL+26n6712PgJ+yiv/bVyOGF2XdkQkoN+j8K1AsA57VRZbf1SgsixxddZ2H9YN/l+fdj7STszLcFcd44BmWrz7pe8P+MNEHOVv8xwvB8EAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADg3PQepePGgN2MSDQAAAAASUVORK5CYII=>