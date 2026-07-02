---
# Ensure that this title is the same as the one in `myst.yml`
title: "Vibes, meet rigor: Evaluating and improving AI performance on complex scientific code"
abstract: |
  Scientists apply rigorous methods to their research, but rarely to the AI tools they use to write code. We tested different large language models (LLMs) in combination with domain-specific tools (including MCP servers and custom-written skills) to find the optimal combination for writing complex domain-specific code. To evaluate LLMs' understanding of Starsim, a Python-based disease modeling framework, we wrote a quantitative proficiency exam. Using this exam, we tested different LLMs and varied their access to tools. Scores ranged widely, from ~50% for Claude Haiku 4.5 and GPT-5 mini without tools to nearly 100% for Claude Opus 4.8 with tools. While LLM choice had the biggest impact on exam scores, tool access also significantly improved performance for both the cheapest (Haiku) and most capable (Opus) models, with more modest benefits for intermediate models. Thus, to improve LLM performance on domain-specific problems, we recommend developing a set of tools with the help of quantitative evaluation.
---

## Introduction

Scientists are often decidedly unscientific about choosing AI tools to help them write code. They know these tools are helpful, but except for trying out different models, they rarely perform controlled evaluations to check whether other changes to their AI workflow produce significantly better results. This is because writing and executing these evaluations is typically time-consuming, the results of the evaluations are difficult to interpret and quantify, and AI workflows and tooling are evolving rapidly. Here we describe our process for quantitatively testing our assumptions about how to build a good AI assistant.

A vast number of benchmarks are available to evaluate the coding ability of LLMs and agents, ranging from resolving real-world software-engineering issues (SWE-bench; @swebench) to completing open-ended tasks in a terminal environment (Terminal-Bench; @terminalbench) and implementing routines for scientific computing (SciCode; @scicode). These benchmarks have driven rapid progress, but they largely target widely-used, general-purpose libraries that are well represented in model training data. In addition, they are at risk of saturation and contamination from newer models being trained on published solutions [@white2025livebench]. 

We wanted to understand how well LLMs handle specialized scientific software: code that is comparatively rare in training datasets, evolves quickly, and demands domain expertise to use correctly. This is precisely the regime in which most research software lives; yet, by its very nature, it cannot be well captured by centralized, standardized benchmarks.

## Methods

### Starsim

Our team models infectious diseases using [Starsim](https://starsim.org/) [@starsim; @starsim_scipy], a high-performance agent-based modeling library built on NumPy [@numpy], SciPy [@scipy], and Numba [@lam2015numba]. A Starsim model is assembled from composable modules — namely diseases (such as HIV or tuberculosis), transmission networks (such as airborne transmission in workplaces or sexual networks), health interventions (such as vaccines and treatments), and demographics (such as pregnancies, births, and deaths). Agents correspond to indices in structured arrays, which provides performance orders of magnitude faster than a fully object-oriented implementation [@kerr2022python]. 

Starsim has been used to model domains ranging from family planning [@o2023fpsim] and sexually transmitted infections [@stuart2024hpvsim] to COVID [@kerr2021covasim]  and Ebola [@delport2024estimating]. Since the diseases themselves are often very complicated, the Starsim models built to model them can also be very complicated. Starsim also uses a number of methodologically subtle features including a unit-aware time system (rates, probabilities, and durations), a custom implementation of random numbers to provide exact deterministic comparability across scenarios [@starsim_crn], and performance-optimized indexing for tracking agents across births and deaths. Combined with a rapidly evolving API and a relatively small training-data footprint, this presents a challenge to AI tools that compounds the usual difficulties of limited context windows and out-of-date training data.

### Starsim-AI

We created an open-source set of agent tools to improve LLMs' performance when working with Starsim, called [Starsim-AI](https://github.com/starsimhub/starsim_ai). Specifically, we added Model Context Protocol (MCP) servers for both Starsim and [Sciris](https://docs.sciris.org), which is a scientific Python library used widely in the codebase [@sciris]. We also created a set of "skills" for Starsim, which consist of problem-solving and feature-oriented Markdown files. Each skill carries a short natural-language description that the agent uses to decide when the skill is relevant, plus a body that is loaded into context only when triggered.

The skills fall into four groups. The largest is a set of *developer* skills, each covering one Starsim subsystem: getting started, configuring the simulation object, diseases, contact networks, interventions, demographics, probability distributions, calibration, multi-disease connectors, analyzers, the time and units system, random number generation, agent indexing, performance profiling, running and comparing multiple simulations, and building non-standard (e.g. compartmental) models. A second group of *style* skills encodes Starsim's conventions for Python code, testing, documentation, and overall design philosophy, so that generated code reads like code a core developer would write. The remaining skills cover the companion [Sciris](https://docs.sciris.org) utility library and the [STIsim](https://github.com/starsimhub/stisim) extension for sexually transmitted infections. These skills were written by Claude Code using the `skill-creator` [plugin](https://code.claude.com/docs/en/skills), based on the (human-written) Starsim [tutorials](https://docs.starsim.org/tutorials) and [user guide](https://docs.starsim.org/user_guide). The skills were then manually reviewed and revised by Starsim core developers for accuracy and completeness.

### Starsim exam

We created a human-written "Starsim exam" evaluation suite designed to test knowledge about Starsim. Although used here for evaluating LLM performance, it is written in the style of a university final exam paper, and is applicable to both human and AI test-takers. The exam covers five topics:

1. **Basics**: General questions about Starsim, and running simple simulations.
2. **SIS dynamics**: Exploring the dynamical behavior of susceptible-infectious-susceptible disease models.
3. **SIRS models**: Comparing agent-based and compartmental models.
4. **Modules**: Calibrating models to data and implementing multi-disease models.
5. **Advanced**: Profiling and debugging code; understanding time units.

In addition, there is a sixth exam question (innocuously labeled "Miscellaneous"), which tests for "cheating" by including questions answerable only with access to (a) the latest Starsim changelog, (b) the Starsim-AI skills, and (c) the solutions themselves.

The exam is subdivided into roughly 30 individually marked sub-questions worth 300 marks in total, and is designed to take a skilled human approximately three hours. The exam is open-ended and code-centric: most sub-questions require the test-taker to write runnable Starsim code, produce figures, and explain the resulting dynamics in prose, with marks awarded via a detailed marking scheme. Questions span the difficulty range from simple recall ("describe what Starsim is and does") to substantial modeling tasks (an example of which is shown in @fig-exam-question).

```{figure} ../figures/fig0_exam_question.png
:label: fig-exam-question
:width: 80%

Example question from the Starsim exam, testing the ability to implement disease models, extract and analyze results, and understand the tradeoffs of different modeling approaches.
```

### Experimental setup

We used two different experimental setups to ensure there was minimal impact of tooling choices on the results. In both cases, agents wrote their answers as Markdown files with executable Python code blocks. In "pipeline mode", evaluations were run using [Inspect](https://inspect.aisi.org.uk/), an evaluation framework developed by the UK AI Security Institute. Agents were run in sandboxed Docker containers using Inspect's agent orchestration infrastructure. Inspect was also used to scored the agents' output, using the (human-written) marking schema. In "humanoid mode", agents were run directly using the [Claude Agent SDK](https://code.claude.com/docs/en/agent-sdk/overview), which also performed marking based on the schema. While the two approaches were similar and produced comparable results, pipeline mode gave greater control over the agents' environment and choice of LLM (including OpenAI models), while humanoid mode provided easier direct access to the agents' thinking and outputs.

In pipeline mode, we ran evaluations with two OpenAI models (GPT-5.4-mini and GPT-5.5) and three Anthropic models (Haiku 4.5, Sonnet 4.6, and Opus 4.8). We ran each model with three configurations: "chat only" (where the LLM was required to provide a single-turn response, and could not run or iterate on code), "agent only" (multi-turn response with access to a Python environment, but without access to Starsim-AI, web search, or other tools), and "agent + skills" (multi-turn response, with access to Starsim-AI skills and MCP servers, plus regular web search). We found it was not possible to run an agent with web search  access but not skills, since agents were surprisingly effective at using web search to discover the Starsim-AI plugin and use it even when it had not been formally installed.

In humanoid mode, we ran evaluations on the three Anthropic models (Haiku 4.5, Sonnet 4.6, and Opus 4.8). For Sonnet and Opus, we also ran evaluations at three effort levels (low, medium, and high; exploratory evaluations were also run at extra high and max levels, but these had similar performance to high, so were not run further to save tokens). These models were also run with three configurations, but slightly different ones. "Chat only" was not available in this mode, since the Claude Agent SDK uses multi-turn responses by default. "Agent-only" was enforced through prompt instructions rather than through Docker firewalls (cheating attempts, such as using `curl` from Bash and `urllib` from Python, were surprisingly frequent, but were intercepted and blocked). "Agent + skills" was similar to pipeline mode. Humanoid mode also used a fourth configuration, "Skills + nudged", where the system prompt included the following "nudge" to use the plugin, since we discovered from exploratory runs that plugin usage was relatively low:

> You have the `starsim-ai` plugin loaded. It provides a set of Starsim-specific skills (covering e.g. diseases, networks, interventions, calibration, demographics, time and units, indexing, distributions, analyzers, and profiling) plus Context7 documentation lookup. USE THE `starsim-ai` SKILLS WHEREVER THEY ARE RELEVANT: before writing Starsim code for a sub-part, invoke the matching skill to ground your approach in the current Starsim API and idioms, and consult the relevant skill whenever you are unsure of the correct Starsim usage. These skills are authoritative for this version of Starsim; prefer them over your prior assumptions.

Each combination (LLM, configuration, and effort level if in humanoid mode) was run five times to account for stochasticity in both answering and marking.

Answers were graded automatically by LLM-based marking agents rather than by hand, with marked answers point-checked and cross-validated against the marking schema by the human experts who wrote the exam. (Human marking was not possible since each a completed exam was typically about 2500 lines, of which approximately half was executable Python code, resulting in approximately 400,000 lines of output across all experiments.)

## Results

### Performance scales with cost and skills

Unsurprisingly, costlier models perform better. The cheapest models with single-turn ("chat") configurations, GPT-mini-chat (US$0.11) and Haiku-chat (US$0.21), also performed by far the worst (56.6% and 54.7%, respectively). The best-performing model, Opus-nudged (98.7%), was also among the most expensive (US$13.38). These results are shown in @fig-cost-score.

```{figure} ../figures/fig1_cost_vs_score.png
:label: fig-cost-score
:width: 80%

Performance as a function of model cost. Shape shows LLM type, and color shows model configuration. Individual runs are shown as small symbols; large symbols are the mean for that model-configuration combination. Not all models were run with all configurations.
```

Since skills tended to be evoked relatively infrequently without nudging (see below), we will focus on results that included nudging. Skills (with nudging) improved the performance of Haiku running in agent mode by 6.7% (from 68.9% to 75.6%), Sonnet by 2.0% (from 94.1% to 96.1%), and Opus by 1.7% (from 97.0% to 98.7%). While these gains may seem small, another way of looking at it is that skills reduced Opus' *mistakes* from 3.0% to 1.3% -- a 57% improvement.

### Model effort quickly saturates

Claude models include a configurable "effort" parameter; higher effort consumes more tokens but is supposed to produce better results. As shown in @fig-effort-score, performance saturated above medium effort (high and extra high effort levels were also run for a subset of trials, which also showed no further improvement; results not shown). Interestingly, skills (with nudging) improved performance in both the lowest-performance case (Sonnet-low, 92.0→95.1%) and the highest-performance cases (Opus-high, 98.3→99.4%) cases, but less in moderate-performance cases (Sonnet-medium, 96.7→96.1%; Opus-low 97.4→97.4%).

```{figure} ../figures/fig2_effort_vs_score.png
:label: fig-effort-score
:width: 80%

Performance as a function of effort for Sonnet and Opus models. Note that vertical bars show minimum and maximum scores, not confidence intervals.
```

### Marks are lost for multiple reasons

No single factor accounted for a majority of marks lost. As shown in @fig-lost-marks, marks were lost approximately equally due to the answer being (a) incorrect, (b) omitted, or (c) for a different reason. Interestingly, the primary improvement in marks was due to a reduction in the number of omitted answers, rather than incorrect answers. This suggests that skills can be helpful in prompting the LLM to bring the solution to completion when it may have otherwise become stuck.

```{figure} ../figures/fig3_lost_marks.png
:label: fig-lost-marks
:width: 80%

Exam marks lost as a function of reason.
```

### The grading is robust to judge choice

Because the answers are graded by LLMs, a natural concern is whether the results are an artifact of a particular judge. @fig-judge addresses this by plotting the Anthropic judge's score against the OpenAI judge's score for every graded answer. The two judges agree closely (Pearson $R^2$ = 0.96) with only a small mean difference (the Anthropic judge is higher by 2.2% on average). The high cross-provider correlation indicates that the scores reflect properties of the answers rather than the idiosyncrasies of a single grader, and gives us confidence in the comparisons above.

That said, the panel does reveal a small but consistent own-provider bias: each judge scores answers written by its own provider's models slightly more generously. On average, there was a preference of 2.1% for a judge towards its own work versus the other provider's. While much smaller than the variance in scores (in one extreme case, there was a 40-point gap between the Anthropic and OpenAI judges' scores on the same exame), it is still notable since even with a rigorous and explicit marking schema, it shows that a model's "personal" bias is still present.

```{figure} ../figures/fig4_judge_agreement.png
:label: fig-judge-agreement
:width: 100%

**(a)** Agreement between the two independent rubric judges. Each point is one graded answer, plotting the Anthropic judge's score (x) against the OpenAI judge's score (y); the dashed line is perfect agreement. The judges agree closely, across both agent configurations (color) and model providers (marker). **(b)** Distribution of score differences between the two judges. An unbiased judge would have a distribution centered on 0; as it is, the Claude judge has a slight preference for Claude models (shifting the distribution to the right), while the Anthropic judge is the opposite (shifting to the left).
```

### Agents were reluctant to use skills

TBC

```{figure} ../figures/fig5_skill_utilization_heatmap.png
:label: fig-skill-utilization
:width: 100%

TBC
```



## Discussion

Our central finding is that, for domain-specific scientific code, *how* a model is equipped can matter as much as *which* model it is. Across the range we tested, giving a model the Starsim-AI skills produced a larger improvement than upgrading to a more capable (and more expensive) base model: a cheap model with tools generally beat an expensive model without them. The practical implication is encouraging for research groups on a budget — a great deal of domain-specific performance can be bought by investing in good tooling around an affordable model, rather than by paying for the largest model available.

The reason tool access helps so much is specific to the regime we are studying. Specialized libraries like Starsim are sparsely represented in training data, evolve quickly, and have idiosyncratic APIs that are easy to use subtly incorrectly. A model relying on its weights alone is therefore working from a stale and incomplete picture. The skills close this gap by surfacing current, authoritative documentation exactly when it is relevant. The integrity question (q05) makes the mechanism vivid: with no access to current information, every model scored zero, because the honest answer is unknowable from training data and the rubric does not reward confident fabrication. Once the skills were available, scores jumped. The benefit of tools is thus not just "more compute" but grounding — replacing plausible guesses with verifiable facts.

These gains are not unconditional, and two boundaries are worth emphasizing. First, there is a floor of base capability below which tools cannot compensate: Haiku improved with skills but remained far behind the other models, suggesting the skills help a model apply knowledge it can already reason about rather than conferring reasoning it lacks. Second, the gains come at a cost in runtime (and tokens): the skills arm took roughly two to three times as long per answer, because the agent reads documentation and iterates more. For interactive use this latency is a real consideration, and the right operating point depends on whether a user values speed or accuracy more for a given task.

A recurring theme is the speed of change. Both the models and the tools we evaluated are moving targets — newer model generations narrow the gaps reported here, and the skills themselves are continuously revised as Starsim evolves. This is precisely why we argue for treating evaluation as a repeatable instrument rather than a one-time measurement: a conclusion about which model or which skill is best has a short shelf life, but the *practice* of measuring it does not. A modest, quantitative exam of the kind we describe can be re-run whenever a new model ships or a skill is updated, turning "vibes" about whether a change helped into a number.

Several limitations temper these conclusions. The study covers a single library in a single domain; while we expect the qualitative finding (tools help most where training data is thin) to generalize, the magnitudes will not transfer directly. Our grading relies on LLM judges, which we mitigate with a fixed marking scheme and a balanced two-provider panel; the small own-provider bias we measured is reassuringly an order of magnitude smaller than the effects of interest, but it is a reminder that LLM-graded benchmarks should never rest on a single judge. The exam, though carefully written, is a finite sample of Starsim usage and may not capture the long tail of real modeling workflows. Finally, the skills were initially drafted by an LLM and then reviewed by core developers; this human-in-the-loop curation was essential to their quality, and an unreviewed, auto-generated skill set might perform very differently.

## Conclusion

Scientists routinely demand rigor of their models but rarely apply it to the AI tools they use to build them. We have shown that a lightweight, exam-style benchmark can quantify the effect of changes to an AI workflow, and that for a complex, fast-moving scientific library, investing in domain-specific tools is at least as effective as moving to a larger model — often allowing a cheaper model to approach the performance of a far more expensive one. We therefore recommend that groups building AI assistants for specialized software pair their tool development with continuous, quantitative evaluation, so that decisions about models and tooling rest on measured improvement rather than intuition.
