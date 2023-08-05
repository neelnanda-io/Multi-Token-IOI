# %%
import os
os.environ["TRANSFORMERS_CACHE"] = "/workspace/cache/"
# %%
from neel.imports import *
from neel_plotly import *
# %%
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.set_grad_enabled(True)
# %%
model: HookedTransformer = HookedTransformer.from_pretrained("gpt2-small")
# %%
n_layers = model.cfg.n_layers
d_model = model.cfg.d_model
n_heads = model.cfg.n_heads
d_head = model.cfg.d_head
d_mlp = model.cfg.d_mlp
d_vocab = model.cfg.d_vocab
# %%
prompt = "When Jemima and John went to the store, John gave a bottle of milk to"
answer = "Jemima"
utils.test_prompt(prompt, answer, model)
# %%
logits, cache = model.run_with_cache(prompt+" "+answer)

stack, labels = cache.get_full_resid_decomposition(expand_neurons=False, apply_ln=True, pos_slice=[-2, -1], return_labels=True)
stack = stack.squeeze(1)
stack.shape
# %%
unembed_dirs = model.W_U[:, model.to_tokens(" Jemima", prepend_bos=False).squeeze(0)].T
unembed_dirs.shape
# %%
line((stack * unembed_dirs).sum(-1).T, x=labels, line_labels=model.to_str_tokens(" Jemima", prepend_bos=False))
# %%
corr_prompts = ["When Jemima and Jemion went to the store, Jemima gave a bottle of milk to Jem", "When Jemima and Jemion went to the store, Jemion gave a bottle of milk to Jem"]
corr_answers = ["ion", "ima"]
utils.test_prompt(corr_prompts[0], corr_answers[0], model, prepend_space_to_answer=False)
utils.test_prompt(corr_prompts[1], corr_answers[1], model, prepend_space_to_answer=False)
# %%
corr_logits, corr_cache = model.run_with_cache(corr_prompts)
answer_tokens = model.to_tokens(corr_answers, prepend_bos=False).squeeze(-1)


corr_stack, labels = corr_cache.get_full_resid_decomposition(expand_neurons=False, apply_ln=True, pos_slice=-1, return_labels=True)
print(f"{corr_stack.shape=}")

corr_unembed_dirs = model.W_U[:, answer_tokens].T
print(f"{corr_unembed_dirs.shape=}")
logit_diff_dir = corr_unembed_dirs[0] - corr_unembed_dirs[1]
corr_unembed_dirs = torch.cat([corr_unembed_dirs, logit_diff_dir.unsqueeze(0)], dim=0)

line((corr_stack @ corr_unembed_dirs.T).reshape(-1, 6).T, x=labels, title="LDA for flipped second token", line_labels=["clean_clean", "clean_corr", "clean_diff", "corr_clean", "corr_corr", "corr_diff"])
head = 9
layer = 9
imshow(corr_cache["pattern", layer][:, head], facet_col=0, x=nutils.process_tokens_index(model.to_tokens(corr_prompts[0])[0], model=model), y=nutils.process_tokens_index(model.to_tokens(corr_prompts[0])[0], model=model))
# %%
line(torch.cat([(stack * unembed_dirs).sum(-1).T, (corr_stack * corr_unembed_dirs).sum(-1).T], dim=0), x=labels, line_labels=model.to_str_tokens(" Jemima", prepend_bos=False) + model.to_str_tokens(" "+corr_answer, prepend_bos=False))

# %%
dlas = torch.cat([(stack * unembed_dirs).sum(-1).T, (corr_stack * corr_unembed_dirs).sum(-1).T], dim=0)
str_tokens = model.to_str_tokens(" Jemima", prepend_bos=False) + model.to_str_tokens(" "+corr_answer, prepend_bos=False)
i1 = 1
i2 = 3
scatter(x=dlas[i1], y=dlas[i2], xaxis=str_tokens[i1], yaxis=str_tokens[i2], hover=labels, color=["H" in l for l in labels])
# %%
v_stack = cache.stack_activation("v").squeeze(1)[:, :-2, :, :]
dla_dir = unembed_dirs[0] / cache["scale"][0, -3, 0]
attn_stack = cache.stack_activation("pattern").squeeze(1)[:, :, -3, :-2]
v_stack.shape, attn_stack.shape, model.W_O.shape, dla_dir.shape
# %%
dla_by_src_pos = einops.einsum(v_stack, attn_stack, model.W_O, dla_dir, "layer pos head d_head, layer head pos, layer head d_head d_model, d_model -> layer head pos")
imshow(dla_by_src_pos.reshape(-1, dla_by_src_pos.shape[-1]), x=model.to_str_tokens(prompt), y=model.all_head_labels())
# %%
