from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence, Union

from .prompts import SampleShot, build_rationale_prompt


@dataclass(frozen=True)
class HFRationaleGeneratorConfig:
    """Configuration for a Hugging Face rationale generator."""

    model_name_or_path: str
    sample_shots: Sequence[SampleShot]
    domain: str = "legal, financial, scientific, or policy"
    num_rationales: int = 8
    include_flag_instructions: bool = True
    max_new_tokens: int = 768
    do_sample: bool = True
    temperature: float = 0.7
    top_p: Optional[float] = None
    torch_dtype: Optional[str] = None
    device_map: Optional[Union[str, Mapping[str, Any]]] = "auto"
    device: Optional[str] = None
    trust_remote_code: bool = False


class HFRationaleGenerator:
    """Callable rationale generator backed by a normal or fine-tuned HF model path."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        sample_shots: Sequence[SampleShot],
        domain: str = "legal, financial, scientific, or policy",
        num_rationales: int = 8,
        include_flag_instructions: bool = True,
        max_new_tokens: int = 768,
        do_sample: bool = True,
        temperature: float = 0.7,
        top_p: Optional[float] = None,
        torch_dtype: Optional[str] = None,
        device_map: Optional[Union[str, Mapping[str, Any]]] = "auto",
        device: Optional[str] = None,
        trust_remote_code: bool = False,
        model: Any = None,
        tokenizer: Any = None,
        generate_fn: Optional[Callable[[str], str]] = None,
    ) -> None:
        self.config = HFRationaleGeneratorConfig(
            model_name_or_path=model_name_or_path,
            sample_shots=sample_shots,
            domain=domain,
            num_rationales=num_rationales,
            include_flag_instructions=include_flag_instructions,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            torch_dtype=torch_dtype,
            device_map=device_map,
            device=device,
            trust_remote_code=trust_remote_code,
        )
        self.generate_fn = generate_fn
        self.model = model
        self.tokenizer = tokenizer
        if self.generate_fn is None and (self.model is None or self.tokenizer is None):
            self.model, self.tokenizer = self._load_model_and_tokenizer()

    def __call__(self, query: str, documents: Optional[Sequence[str]] = None) -> str:
        prompt = self.build_prompt(query)
        if self.generate_fn is not None:
            return self.generate_fn(prompt)
        return self._generate_with_hf(prompt)

    def build_prompt(self, query: str) -> str:
        return build_rationale_prompt(
            query,
            sample_shots=self.config.sample_shots,
            domain=self.config.domain,
            num_rationales=self.config.num_rationales,
            include_flag_instructions=self.config.include_flag_instructions,
        )

    def _load_model_and_tokenizer(self):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError("Install METEORA with the hf extra: pip install -e '.[hf]'") from exc

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name_or_path,
            trust_remote_code=self.config.trust_remote_code,
        )
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs = {"trust_remote_code": self.config.trust_remote_code}
        if self.config.device_map is not None:
            model_kwargs["device_map"] = self.config.device_map
        dtype = _resolve_torch_dtype(torch, self.config.torch_dtype)
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype
        model = AutoModelForCausalLM.from_pretrained(self.config.model_name_or_path, **model_kwargs)
        if self.config.device and hasattr(model, "to"):
            model = model.to(self.config.device)
        return model, tokenizer

    def _generate_with_hf(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        target_device = self.config.device or _infer_model_device(self.model)
        if target_device is not None and hasattr(inputs, "to"):
            inputs = inputs.to(target_device)

        generation_kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": self.config.do_sample,
            "temperature": self.config.temperature,
        }
        if self.config.top_p is not None:
            generation_kwargs["top_p"] = self.config.top_p

        outputs = self.model.generate(**inputs, **generation_kwargs)
        output = outputs[0] if hasattr(outputs, "__getitem__") else outputs
        try:
            input_length = inputs["input_ids"].shape[-1]
            output = output[input_length:]
        except (KeyError, TypeError, AttributeError, IndexError):
            pass
        return self.tokenizer.decode(output, skip_special_tokens=True).strip()


def _resolve_torch_dtype(torch_module: Any, dtype_name: Optional[str]) -> Any:
    if dtype_name is None:
        return None
    if dtype_name == "auto":
        return "auto"
    dtype = getattr(torch_module, dtype_name, None)
    if dtype is None:
        raise ValueError(f"Unknown torch dtype: {dtype_name}")
    return dtype


def _infer_model_device(model: Any) -> Any:
    device = getattr(model, "device", None)
    if device is not None and getattr(device, "type", str(device)) != "meta":
        return device
    try:
        first_parameter = next(model.parameters())
    except (AttributeError, StopIteration, TypeError):
        return None
    parameter_device = getattr(first_parameter, "device", None)
    if parameter_device is not None and getattr(parameter_device, "type", str(parameter_device)) != "meta":
        return parameter_device
    return None
