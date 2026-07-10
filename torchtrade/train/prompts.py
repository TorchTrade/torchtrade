"""Build training prompts identical to inference prompts (reuse BaseLLMActor)."""


def build_training_prompt(actor, tensordict, system_prompt=None, user_prompt_fn=None):
    """Return (system_prompt, user_prompt) for one bar, honoring the same override
    hooks the actor uses at inference (so training prompts == inference prompts).

    system_prompt: None -> actor._resolve_system_prompt(); str -> used as-is;
        callable f(actor) -> str.
    user_prompt_fn: None -> actor._construct_user_prompt(td); else f(actor, td) -> str.
    """
    if system_prompt is None:
        sysp = actor._resolve_system_prompt()
    elif callable(system_prompt):
        sysp = system_prompt(actor)
    else:
        sysp = system_prompt

    userp = user_prompt_fn(actor, tensordict) if user_prompt_fn is not None \
        else actor._construct_user_prompt(tensordict)
    return sysp, userp
