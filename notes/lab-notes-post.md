# Fine-tuning a small model to drive Nushell: lab notes

This is the first in a series of notebooks I'll be posting about fine-tuning small models.

Here are my rough notes on what went well and what didn't go well.

> **Claude:** I'm Claude, the coding agent Max built this with. My notes are the indented ones. I've kept them to numbers from the saved runs.

## Went well

Synthetic data worked great, but what worked better was to treat it as an inverse problem. What I mean by that is, you should start from the endpoint you're trying to achieve (in my case, the successful command execution) and work backwards. This will let you know precisely when and where things go wrong.

> **Claude:** Concretely: a call counts only if running it returns the reference value. On 231 requests from four projects the model never saw, it got 214 (92.6%). The base model got 3 of the 40 it was run on.

You should also spend most of your time on the data generation itself. My first couple of passes at the data were too naive to transfer usefully. What worked was setting up a rubric and asking:

* For the environment
   * What are all the commands we have access to?
      * How will they all combine?
   * What can go wrong?
      * Parsing, wrong paths, etc. I handled these by focusing on command generation that didn't need hardcoded strings.
      * What can we detect automatically when things go wrong?
      * What can we grade when things compile but maybe still aren't correct?
   * Where do we want to make simplifications in the first version?
      * I put data and SQL out of scope for this first version, as well as agents.
* For the user
   * What will they ask?
   * What tone will they take?
   * Are there metaphors or analogies they'll use? Shorthand?
   * With what frequency do we think they'll call which commands?

Set these up as cross-tabs ahead of time and be ready to read into the individual rows.

> **Claude:** The one cross-tab we didn't make cost us. Category shares came from usage data, but nothing counted chains per kind of file. The first run had 5 training chains that open a JSON file, and 4 of its 11 wrong calls were `open --raw` on JSON.

I strongly recommend literate programming and interactivity. With agents it's harder than ever to really understand what is being built unless you spend a lot of time on it. So in the notebook, I had some of the core sections exposed as faders and knobs, used live edit, put charts in there, and made this great 3D visualization to make sure I really understood a little bit more of what was going on, and also to make it more communicable to other people in the future.

> **Claude:** From my side of that: the data generator runs line by line on the page for whichever environment is picked, so Max could read what I had wired up without opening the code.

You probably need more data than you think. Even at 20 environments with hundreds of commands, I could have scaled way up and gotten additional performance and generality out of this same training.

> **Claude:** The two runs we have: 72% with 162 chains in 10 environments, 92% with 401 chains in 20. They used different held-out projects and settings, so it's two points, not a curve.

The LoRA settings suggested in the Liquid notebook, along with a reading of "LoRA Without Regret" by John Schulman at Thinking Machines, basically provided the settings, and they worked great.

> **Claude:** What the sweep picked: rank 8 on every linear layer, learning rate 2e-4. It was scored on 24 requests, so only the worst setting (lr 4e-4, 42%) is clearly apart from the best (75%). The final adapter is 8.2 M parameters, trained on 825 rows for 2 epochs.

Finally, starting with a single case and overfitting to it is always a really great idea to prove your learning setup works at all, and then scaling up slowly. It takes a bit of time, but it's helped me detect when settings are wrong or when the agent doing the coding wired up slop.

> **Claude:** Fair. One example from this project: my first check on rewritten requests accepted "document" as keeping the folder name `doc`, so bare folder names were getting lost. Reading the rows caught it; the scores didn't.

## What didn't go well

I started with a fancier learning setup I didn't need. I tried to train a grader with D1 and use guidance. My plan had been to get an additional couple of points of performance. But the LoRA training was so strong that, with my size of data, the guidance itself became mathematically impossible to use. So until you really understand the cuts of your data and what's generating the learning, don't reach for fancier stuff yet.

> **Claude:** The numbers behind "impossible": along the reference calls, the model's first-choice token led its runner-up by a median of 11.1 nats, about 66,000 to 1. Guidance could only pick a token within 2.3 nats of the top, which left it a choice at 19 of 1,290 tokens. No guided decoder beat the plain adapter (89.6–92.6% against 92.6%). The grader itself trained fine: 96% on failure type.

I didn't spend enough time on the data or generate enough. If I'd had a lot more, I could have figured out a smarter training setting, I think. I also didn't try to filter down on subsets beforehand to see where the learning signal was coming from. This doesn't need to be terribly complicated either: reading and using the system a bit goes a long way. This was a proof of concept, and now that I have a working model, that's where I'd go next.

I should have also spent way more time on the failure cases. Moving forward, I'd probably define that up front for all of the important cuts of my data. I think I left some performance on the table just because of the data itself.

> **Claude:** I read all 17 wrong calls. 6 leave out a step the request names, 5 write a different step, and 6 are requests that don't contain what the reference needs (the model that wrote the requests dropped a key name or repeated its instructions). Leaving those 6 out, it's 214 of 225 (95%). The 11 real errors come from 5 chain patterns.
