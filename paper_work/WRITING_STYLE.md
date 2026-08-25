# Writing guidelines: avoiding generative-AI style

Distilled from the edits made to the c107fa interim report. Applies to reports,
proposals and papers. The underlying principle is that AI prose is
over-organised: it constantly tells the reader what is coming, what just
happened, and how to feel about it. Technical writing states things and moves
on.

## 1. Punctuation and connectives

**Em dashes.** Do not use them. Almost every em dash is a comma, a semicolon, a
full stop or a parenthesis. If a clause needs an em dash to attach, it usually
wants to be its own sentence.

**"Rather than."** Avoid entirely. It is grammatical English, but AI reaches for
it as the default contrast and the frequency is the tell. Use "and not",
"instead of", "as opposed to", or restructure so no contrast is needed.

- ✗ The shortfall was caused by an under-specified estimate rather than by
  inefficient code.
- ✓ The shortfall was caused by an under-specified estimate. The code was not
  inefficient.

**"Not X but Y."** Same problem, same fix.

**Colons introducing an explanation.** Fine occasionally, exhausting at three
per page. Count them.

## 2. Do not announce, just say

The strongest single tell. A paragraph opens with a short abstract sentence
declaring its own topic, and the actual content arrives in sentence two.

- ✗ This has a direct effect on the figures given above. At the conversion rate
  implied by the centre's proposal, the unused H100 allocation corresponds to
  about 34,500 A100 GPU-hours.
- ✓ At the conversion rate implied by the centre's proposal, the unused H100
  allocation corresponds to about 34,500 A100 GPU-hours.

The announcing sentence is almost always deletable. Test: remove the first
sentence of each paragraph. If nothing is lost, it was scaffolding.

Related pattern: **enumerating before enumerating.**

- ✗ Two costs were omitted from the estimate. The first is the teacher forward
  pass...
- ✓ The teacher forward pass was left out of the estimate altogether...
  Exploratory compute was likewise not budgeted...

Also: **answering a section heading as though it were a question from a
reader.**

- ✗ Were the approved resources sufficient? No. The allocation was granted
  for...
- ✓ Were the approved resources sufficient? The allocation was granted for the
  period from 03.02.2026 to 31.01.2027 and was fully consumed by August 2026, so
  the twelve-month budget was spent in about six months. The approved resources
  were therefore not sufficient.

Put the conclusion after the evidence, inside a sentence, not alone at the
front.

## 3. Do not rate your own material

Sentences whose only function is to tell the reader that the previous sentence
was important.

Banned or heavily rationed: *telling, instructive, notable, notably, striking,
remarkable, importantly, significantly, crucially, it is worth noting, worth
naming, the most consequential, the most costly, this shows that, this
demonstrates that, this underscores, what is striking here.*

- ✗ The results of this phase were stronger than we had expected.
- ✓ [delete; state the result and let the number carry it]

- ✗ We mention it here because it shows that the shortfall was not caused by
  inefficient use of the machines.
- ✓ The shortfall described above therefore did not arise from inefficient use
  of the machines.

Replace adjectives with numbers wherever possible. "The greater part of the
requested budget" became "approximately three quarters of the resources
requested in the application." A number is never AI-flavoured.

## 4. No rhetorical payoffs

AI writes paragraph-closing sentences that are shaped. Chiasmus, tricolon,
antithesis, aphorism. Human technical writing ends when the information ends.

- ✗ The estimate reserved its budget for the experiments we did not perform, and
  left unbudgeted the experiments we did.
- ✓ The budget had in effect been assigned to the wrong category of work.

- ✗ Both concentrate the training signal a small model receives per token, one
  by supplying a teacher's output distribution, the other by discarding
  low-signal text.
- ✓ ...the former through the output distribution of a teacher model and the
  latter through the removal of low-signal text.

The second is still a balanced pair, but it is a plain "former/latter"
construction and not a built parallelism with a colon setting it up.

## 5. No meta-commentary about the document

Do not describe what the document is doing, or characterise your own candour.

- ✗ ...and we should set out plainly why.
- ✗ Two omissions should be named explicitly, since both can be avoided in
  future.
- ✗ We record this as a change of direction and not as an unfinished work
  package.

The last one survived in the report because it does real work: it tells the
reader how to read a line item. Most such sentences do not.

## 6. Rhythm

AI produces sentences of unnervingly even length, three to five per paragraph,
each 18–22 words. Vary deliberately. Target a standard deviation of sentence
length above roughly 9 words. Let some sentences run to 40 words with proper
subordination, and let others be six.

Report register specifically:

- more passive voice than an essay ("the classifiers were applied to...")
- more nominalisation ("the reduction of the data required" over "reducing the
  data required")
- longer subordinated sentences, fewer punchy ones
- numbered lists where a list exists; do not prose out an enumeration

## 7. Vocabulary to watch

Words that are fine individually and damning in aggregate: *genuinely,
substantially, considerably, comparatively, straightforwardly, effectively,
essentially, fundamentally, precisely, plainly, durable, robust, meaningful,
key, leverage, delve, landscape, realm, testament, underscore.*

Also: *"In practice", "In effect", "That said", "To be clear", "It is worth
remembering".*

## 8. Checklist before sending

1. Count em dashes. Should be zero.
2. Search "rather than". Should be zero.
3. Delete the first sentence of every paragraph. Restore only those whose loss
   hurts.
4. Search the significance vocabulary in §3 and §7.
5. Read the last sentence of each paragraph aloud. If it sounds like a closing
   line, cut or flatten it.
6. Check sentence-length variance. If every sentence is medium, rewrite some
   long and some short.
7. Replace every evaluative adjective with a number where a number exists.

## 9. What not to overcorrect

Do not manufacture bad English. The aim is a competent researcher writing an
administrative document, not a non-native speaker struggling. Keep:

- attributed causes with hedges ("which we attribute to the weaker annotator
  models")
- genuine list introductions ("The following measures have been introduced for
  the follow-up project.")
- short sentences that carry content ("The H100 share was not used.")
- standard scientific connectives ("therefore", "however", "in addition")

The target is flatness of delivery, not clumsiness of language.
