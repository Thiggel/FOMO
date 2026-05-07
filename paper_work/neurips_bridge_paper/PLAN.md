# NeurIPS Paper Revision Plan

## Goal
Address the current reviewer/internal comments with a focused manuscript pass. The priority is to make the method section feel complete, clarify how image-to-image generation is performed, add only the citation coverage that is actually missing, and make every linear-probe versus kNN result easy to identify in the prose.

## Highest-priority edits

1. Tighten Sections 1 and 2 citation coverage without turning them into a literature survey.
   - Check whether DINOv3 should be cited alongside DINO and DINOv2 in the opening SSL context.
   - Add at most one compact DINOv3 citation if used; avoid broadening the paragraph with many new SSL papers.
   - Current likely edit locations:
     - Introduction first paragraph: `bridge_neurips_2026.tex`, around the sentence listing contrastive learning, masked prediction, and self-distillation.
     - Related Work, self-supervised visual representation learning paragraph.
   - Add a `references.bib` entry for DINOv3 if the citation is added.
   - Verified candidate: DINOv3, arXiv:2508.10104, "DINOv3: Self-supervised learning for vision at unprecedented scale".

2. Expand Section 3 so the method reads like a full proposal.
   - Add a short bridge paragraph at the start of Section 3 explaining the loop as three reusable operators: score, select, repair.
   - Make the interface explicit:
     - inputs: unlabeled source set, SSL encoder, OOD score hyperparameters, generation budget, frozen image-to-image generator;
     - outputs per cycle: repaired source set and next-cycle encoder.
   - Add a compact "what is optimized versus frozen" clarification: SSL encoder is trained; generator is frozen; selection is nonparametric.
   - Keep this expansion concise, roughly 0.25-0.4 pages, so it strengthens the method without squeezing results.

3. Clarify Section 3.3 generation / translation mechanics.
   - Rename or revise `Diffusion based repair` to make it explicitly image-to-image, for example "Image-to-image diffusion repair".
   - State how each selected anchor image is translated:
     - selected image is used as the conditioning input;
     - generation produces `G` variants per anchor;
     - model is frozen;
     - SD3 is default and FLUX is used only in the ablation;
     - no-generation re-population re-adds originally removed/held-out source images without diffusion.
   - Add implementation details that matter for reproducibility if known: prompt policy, strength/noise level, resolution, random seed/noise sampling, filtering or no filtering. If these details are only in code, inspect the generation script before writing.
   - Add paired qualitative examples: input anchor -> SD3 output and input anchor -> FLUX output if available.
   - Reuse `figures/generated_samples_overview.png` if it already shows paired inputs and outputs. If not, create a new compact figure such as `figures/generation_pairs.pdf` with 3-4 anchor rows and columns for input, SD3, and FLUX.
   - Best placement: end of Section 3.3 if space permits; otherwise place the full figure in the appendix and include one sentence in Section 3.3 pointing to it.

4. Keep Section 4 statistical and ablation-protocol strengths.
   - Preserve the existing sentence: "All reported numbers are mean ± standard deviation over three seeds."
   - Preserve the generation ablation explanation comparing SD3, FLUX, and no-generation re-population.
   - Only revise for clarity if needed after Section 3.3 is expanded, to avoid duplicate wording.

5. Revise Section 5 result references so evaluation type is explicit every time.
   - Update prose to name both the table and evaluation protocol in the sentence.
   - Target wording pattern:
     - "Table 1 (linear-probe) and Table 2 (kNN) are ..."
     - "Table 3 (linear-probe) reports ..."
     - "Table 4 (kNN) shows ..."
     - "Table 5 (linear-probe) evaluates ..."
   - Apply the same style to source-regime and ablation references currently phrased as `Table~... reports` or `Table~... shows`.
   - Check the final compiled table numbering before finalizing, because current source order includes separate label-source, web-source, and combined-ablation tables.

## Detailed section plan

### Section 1: Introduction
- Keep the current motivation: SSL inherits data coverage biases.
- Add DINOv3 only if it naturally fits the single sentence listing modern SSL families.
- Do not add a paragraph of new related work here.
- Add one sentence, if space allows, previewing that BRIDGE repairs data through image-to-image generation rather than text-prompted generation from scratch.

### Section 2: Related Work
- Keep four short paragraphs.
- In the SSL paragraph, mention DINOv3 as a scaling-era example only if cited in Section 1.
- In the diffusion augmentation paragraph, sharpen the contrast:
  - prior work often uses generation as augmentation;
  - BRIDGE uses selected real images as anchors and performs targeted image-to-image repair inside the SSL loop.
- Avoid adding many supervised long-tail generation papers unless a specific missing citation is identified.

### Section 3: Method
- Add a method overview paragraph before Section 3.1.
- Section 3.1: keep equations but add a sentence explaining why mean kNN distance is compatible with unlabeled data.
- Section 3.2: keep mode-window equations; consider adding one sentence defining why `q75--q99` is a default rather than a tuned-per-task value.
- Section 3.3: expand generation mechanics and add/point to paired examples.
- Section 3.4: keep design rationale, but remove any overlap created by the new Section 3 overview.

### Section 4: Experimental Setup
- Keep the three-seed reporting statement.
- Keep the ablation protocol sentence about SD3, FLUX, and no-generation re-population.
- If Section 3.3 defines no-generation re-population, Section 4 can remain short and refer to it without re-explaining.

### Section 5: Results
- Audit every table reference for protocol labels: linear-probe, kNN, or combined.
- Split ambiguous sentences where both linear-probe and kNN are discussed together.
- Fix one apparent inconsistency in the generation paragraph:
  - current text says SD3 is 41.4, FLUX is 35.0, no-generation is 32.6, then says no-generation is on par with diffusion and outperforms FLUX.
  - The plan should resolve this before editing: either the table values changed, or the paragraph is stale. The final prose must match the table.
- Maintain the strong existing point that kNN corroborates feature-geometry gains rather than only linear-head gains.

## Asset plan

- Inspect `figures/generated_samples_overview.png`.
- If it already contains clear paired input-output examples:
  - add it to the appendix or Section 3.3 with a caption emphasizing anchor image, SD3 translation, and FLUX translation.
- If it does not:
  - generate a new figure from saved generation outputs with columns: input anchor, SD3 output, FLUX output, optional no-generation/re-populated example.
  - use only a few rows so it is readable in the main paper.
- Caption should answer the reviewer question directly: the selected input image conditions the image-to-image generator, producing variants that preserve coarse semantics while altering appearance/composition.

## Verification checklist

- Compile the paper after edits and verify no missing citations, missing figure files, or unresolved references.
- Check final table numbering in the PDF and update Section 5 prose accordingly.
- Confirm Section 3 length increased but the main paper remains within the NeurIPS page budget.
- Confirm all table captions or surrounding sentences clearly distinguish linear-probe from kNN.
- Confirm the generation ablation prose matches actual table values.

## Suggested order of work

1. Inspect generation code/configs to recover exact SD3/FLUX image-to-image parameters.
2. Inspect `generated_samples_overview.png` and decide whether it satisfies paired-example needs.
3. Add or prepare the qualitative figure.
4. Update Sections 1-3.
5. Update Section 5 table-reference language and fix the generation-ablation inconsistency.
6. Add DINOv3 bib entry if used.
7. Compile and do a final PDF pass.
