import time
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from langchain.callbacks.streaming_stdout_final_only import FinalStreamingStdOutCallbackHandler
from langchain.schema import LLMResult
from uuid import UUID

class CustomStreamingStdOutCallbackHandler(FinalStreamingStdOutCallbackHandler):

	buffer: List[Tuple[str, float]] = []
	stop_token = "#!stop!#"
		
	def on_llm_start(
		self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
	) -> None:
		super().on_llm_start(serialized, prompts, **kwargs)
		self.buffer = []
		
	def on_llm_end(
		self,
		response: LLMResult,
		*,
		run_id: UUID,
		parent_run_id: Optional[UUID] = None,
		**kwargs: Any,
	) -> Any:
		self.add_to_buffer(self.stop_token)
	
	def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
		# Remember the last n tokens, where n = len(answer_prefix_tokens)
		self.last_tokens.append(token)
		if len(self.last_tokens) > len(self.answer_prefix_tokens):
			self.last_tokens.pop(0)

		# Check if the last n tokens match the answer_prefix_tokens list ...
		if self.last_tokens == self.answer_prefix_tokens:
			self.answer_reached = True
			# Do not print the last token in answer_prefix_tokens,
			# as it's not part of the answer yet
			return

		# ... if yes, then append tokens to buffer
		if self.answer_reached:
			self.add_to_buffer(token)
			
	def add_to_buffer(self, token:str) -> None:
		now = datetime.now()
		self.buffer.append((token, now))

	def stream_chars(self):
		while True:
			# when we didn't receive any token yet, just continue
			if len(self.buffer) == 0:
				continue
			
			token, timestamp = self.buffer.pop(0)
			
			if token != self.stop_token:
				for character in token:
					yield (character, timestamp)
			else:
				break