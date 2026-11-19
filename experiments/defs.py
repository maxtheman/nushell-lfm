"""What the experiment scripts read from the pipeline, in one dictionary. They were written when everything
lived in the notebook and took its definitions as `defs`; this builds the same dictionary from the modules."""


def load_defs(log=print) -> dict:
    from nushell_lfm.calls import parse_tool_call
    from nushell_lfm.core import MODEL_FOLDERS, MODELS_DIR, model_path
    from nushell_lfm.lora import attach_lora, learning_rate, set_lora
    from nushell_lfm.target import TARGET_DIR, score_call, target_prompt

    import guidance
    from data import build_full
    from grader import GRADER_DIR, encode_grader_row, grader_readouts, run_grader_experiments, run_grader_sweep

    built = build_full()
    log(f"dataset {built['dataset']['fingerprint']} (target rows plus broken and equivalent variants for the grader)")
    return {
        **built, "MODEL_FOLDERS": MODEL_FOLDERS, "MODELS_DIR": MODELS_DIR, "TARGET_DIR": TARGET_DIR, "GRADER_DIR": GRADER_DIR,
        "attach_lora": attach_lora, "set_lora": set_lora, "learning_rate": learning_rate, "model_path": model_path,
        "parse_tool_call": parse_tool_call, "score_call": score_call, "target_prompt": target_prompt,
        "encode_grader_row": encode_grader_row, "grader_readouts": grader_readouts,
        "run_grader_experiments": run_grader_experiments, "run_grader_sweep": run_grader_sweep,
        "run_guidance_pipeline": guidance.run_guidance_pipeline, "crossfit_wrong_calls": guidance.crossfit_wrong_calls,
        "train_bad_adapter": guidance.train_bad_adapter, "score_decoders": guidance.score_decoders,
        "score_low_bit": guidance.score_low_bit, "rebuild_weak_target": guidance.rebuild_weak_target,
    }
