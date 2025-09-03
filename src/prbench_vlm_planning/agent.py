"""VLM planning agent for prbench environments."""

from typing import Any, Hashable, List, Optional, TypeVar, cast
import os
import numpy as np
import PIL.Image
from PIL import ImageDraw

from prpl_llm_utils.models import OpenAIModel
from prpl_llm_utils.cache import FilePretrainedLargeModelCache
from pathlib import Path

def create_vlm_by_name(model_name: str):
    """Create a VLM instance using prpl_llm_utils."""
    # Create a cache directory in the current working directory
    cache_dir = Path("./vlm_cache")
    cache_dir.mkdir(exist_ok=True)
    cache = FilePretrainedLargeModelCache(cache_dir)
    
    if "gpt" in model_name.lower() or "claude" in model_name.lower():
        return OpenAIModel(model_name, cache)
    else:
        raise ValueError(f"Unsupported VLM model: {model_name}")
from prpl_utils.gym_agent import Agent


_O = TypeVar("_O", bound=Hashable)
_U = TypeVar("_U", bound=Hashable)

class VLMPlanningAgentFailure(BaseException):
    """Raised when the VLM planning agent fails."""

class VLMPlanningAgent(Agent[_O, _U]):
    """VLM-based planning agent for prbench environments."""
    
    def __init__(
        self,
        vlm_model_name: str = "gpt-4o",
        temperature: float = 0.0,
        max_planning_horizon: int = 50,
        seed: int = 0,
        env_models: Optional[Any] = None,
        use_image: bool = True,
    ) -> None:
        """Initialize the VLM planning agent.
        
        Args:
            vlm_model_name: Name of the VLM model to use
            temperature: Temperature for VLM sampling
            max_planning_horizon: Maximum steps in a plan
            seed: Random seed
            env_models: Optional environment models from prbench_models
            use_image: Whether to use image observations
        """
        super().__init__(seed)
        
        self._vlm_model_name = vlm_model_name
        self._vlm = create_vlm_by_name(vlm_model_name)
        self._seed = seed
        self._temperature = temperature
        self._max_planning_horizon = max_planning_horizon
        self._env_models = env_models
        self._use_image = use_image
        
        # Current plan state
        self._current_plan: Optional[List[_U]] = None
        self._plan_step = 0
        self._last_obs: Optional[_O] = None
        
        # Load base prompt from file
        self._base_prompt = self._load_base_prompt()
        
    def _load_base_prompt(self) -> str:
        """Load the base planning prompt from file."""
        # Get the path to the prompt file
        current_dir = os.path.dirname(__file__)
        prompt_path = os.path.join(current_dir, "prompts", "vlm_planning_prompt.txt")
        
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()

    def reset(self, obs: _O, info: dict[str, Any]) -> None:
        """Reset the agent for a new episode."""
        super().reset(obs, info)
        self._current_plan = None
        self._plan_step = 0
        
        try:
            self._current_plan = self._generate_plan(obs, info)
        except Exception as e:
            raise VLMPlanningAgentFailure(f"Failed to generate initial plan: {e}")

    def _get_action(self) -> _U:
        """Get the next action from the current plan."""
        if not self._current_plan:
            raise VLMPlanningAgentFailure("No current plan available")
            
        if self._plan_step >= len(self._current_plan):
            raise VLMPlanningAgentFailure("Plan exhausted")
            
        action = self._current_plan[self._plan_step]
        self._plan_step += 1
        return action

    def _generate_plan(self, obs: _O, info: dict[str, Any]) -> List[_U]:
        """Generate a plan using the VLM."""
        
        # Store observation for goal derivation
        self._last_obs = obs
        
        # Prepare images if available and using images
        images = None
        if self._use_image and hasattr(obs, 'get') and 'rgb' in obs:
            rgb_obs = obs['rgb']
            if isinstance(rgb_obs, np.ndarray):
                pil_img = PIL.Image.fromarray(rgb_obs)
                # Add text overlay indicating this is the initial state
                draw = ImageDraw.Draw(pil_img)
                text = "Initial state for planning"
                # Simple text overlay at top-left
                draw.text((10, 10), text, fill="white")
                images = [pil_img]
        
        # Prepare prompt context
        actions_str = self._get_available_actions_str(info)
        objects_str = self._get_objects_str(obs, info)
        goal_str = self._get_goal_str(info)
        
        prompt = self._base_prompt.format(
            actions=actions_str,
            objects=objects_str,
            goal=goal_str
        )
        
        # Query VLM
        try:
            # Prepare hyperparameters for prpl_llm_utils
            hyperparameters = {
                "temperature": self._temperature,
                "seed": self._seed,
            }
            
            # Query the VLM
            response = self._vlm.query(
                prompt=prompt,
                imgs=images,
                hyperparameters=hyperparameters
            )
            
            # Extract text from response
            plan_text = response.text
            
            # Parse the plan
            return self._parse_plan_from_text(plan_text, obs, info)
            
        except Exception as e:
            raise VLMPlanningAgentFailure(f"VLM query failed: {e}")


    def _get_available_actions_str(self, info: dict[str, Any]) -> str:
        """Get string description of available actions."""
        if self._env_models is not None:
            # Check if we have environment-specific controller info
            if isinstance(self._env_models, dict) and 'controllers' in self._env_models:
                controllers = self._env_models['controllers']
                actions = []
                for name, controller_class in controllers.items():
                    # Extract parameter space information from controller
                    param_space_str = self._get_controller_param_space(controller_class)
                    object_params_str = self._get_controller_object_params(name)
                    actions.append(f"{name}({object_params_str}), params_space={param_space_str}")
                return "\n".join(actions)
            # Check if it's a full bilevel planning model with skills
            elif hasattr(self._env_models, 'skills'):
                skills = self._env_models.skills
                actions = []
                for skill in skills:
                    actions.append(f"{skill.name}: {skill.description if hasattr(skill, 'description') else 'No description'}")
                return "\n".join(actions)
        
        # No controllers available
        raise ValueError("Environment models with controllers or skills required for action extraction")
    
    def _get_controller_param_space(self, controller_class) -> str:
        """Extract parameter space information from controller class."""
        import inspect
        import typing
        
        try:
            # Get the sample_parameters method signature
            sample_params_method = getattr(controller_class, 'sample_parameters', None)
            if sample_params_method is None:
                return "Box([0.0], [1.0], (1,), float32)"
            
            # Get return type annotation
            sig = inspect.signature(sample_params_method)
            return_annotation = sig.return_annotation
            
            # Parse the return type to determine parameter space
            if return_annotation == inspect.Signature.empty:
                # No annotation, try to infer from method name
                return self._infer_param_space_from_name(controller_class.__name__)
            
            # Handle different return type annotations
            if hasattr(typing, 'get_origin') and typing.get_origin(return_annotation) is tuple:
                # Get tuple arguments
                args = typing.get_args(return_annotation)
                num_params = len(args)
                if num_params > 0:
                    # Most controllers use [0.0, 1.0] range for relative parameters
                    lower_bounds = "[" + ", ".join(["0.0"] * num_params) + "]"
                    upper_bounds = "[" + ", ".join(["1.0"] * num_params) + "]"
                    return f"Box({lower_bounds}, {upper_bounds}, ({num_params},), float32)"
            elif return_annotation == float:
                return "Box([0.0], [1.0], (1,), float32)"
            
            # Fallback to name-based inference
            return self._infer_param_space_from_name(controller_class.__name__)
            
        except Exception:
            # Fallback for any errors
            return "Box([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], (3,), float32)"
    
    def _infer_param_space_from_name(self, controller_name: str) -> str:
        """Infer parameter space based on controller name."""
        if "MoveToTgt" in controller_name or "MoveToPassage" in controller_name:
            return "Box([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], (3,), float32)"
        elif "Grasp" in controller_name:
            return "Box([0.0, 0.0], [1.0, 1.0], (2,), float32)" 
        else:
            return "Box([0.0], [1.0], (1,), float32)"
    
    def _get_controller_object_params(self, controller_name: str) -> str:
        """Get object parameter names based on controller name."""
        if "MoveToTgt" in controller_name:
            return "robot, target"
        elif "MoveToPassage" in controller_name:
            return "robot, obstacle1, obstacle2"
        elif "Grasp" in controller_name:
            return "robot, object"
        elif "Place" in controller_name:
            return "robot, object, surface"
        elif "Push" in controller_name:
            return "robot, object, target"
        else:
            return "robot, ..."

    def _get_objects_str(self, obs: _O, info: dict[str, Any]) -> str:
        """Get string description of objects in the scene."""
        # Require bilevel planning models with observation_to_state
        if self._env_models is None or not hasattr(self._env_models, 'observation_to_state'):
            raise ValueError("Environment models with observation_to_state function required for object extraction")
        
        state = self._env_models.observation_to_state(obs)
        objects = []
        
        # Extract all objects from the state
        if hasattr(state, 'objects'):
            # If state has objects attribute, iterate through them
            for obj in state.objects:
                obj_info = f"{obj.name}:{obj.obj_type.name}"
                # Add position if available
                if hasattr(state, 'get'):
                    try:
                        x = state.get(obj, "x")
                        y = state.get(obj, "y") 
                        obj_info += f" at ({x:.2f}, {y:.2f})"
                    except:
                        pass
                objects.append(obj_info)
        elif hasattr(state, 'data'):
            # Alternative: extract from state data
            for obj_name, obj_props in state.data.items():
                if isinstance(obj_props, dict):
                    # Get key properties like position
                    pos_info = ""
                    if 'x' in obj_props and 'y' in obj_props:
                        pos_info = f" at ({obj_props['x']:.2f}, {obj_props['y']:.2f})"
                    objects.append(f"{obj_name}{pos_info}")
        else:
            raise ValueError("Unable to extract objects from state")
        
        if not objects:
            raise ValueError("No objects found in state")
        
        return "\n".join(objects)

    def _get_goal_str(self, info: dict[str, Any]) -> str:
        """Get string description of the goal."""
        # Require bilevel planning models with goal_deriver
        if (self._env_models is None or 
            not hasattr(self._env_models, 'goal_deriver') or 
            not hasattr(self._env_models, 'observation_to_state')):
            raise ValueError("Environment models with goal_deriver function required for goal extraction")
        
        if not hasattr(self, '_last_obs') or self._last_obs is None:
            raise ValueError("No observation available for goal derivation")
        
        state = self._env_models.observation_to_state(self._last_obs)
        goal_obj = self._env_models.goal_deriver(state)
        
        # Extract goal description from RelationalAbstractGoal
        if hasattr(goal_obj, 'atoms'):
            goal_atoms = []
            for atom in goal_obj.atoms:
                if hasattr(atom, 'predicate') and hasattr(atom, 'objects'):
                    pred_name = atom.predicate.name if hasattr(atom.predicate, 'name') else str(atom.predicate)
                    obj_names = [obj.name if hasattr(obj, 'name') else str(obj) for obj in atom.objects]
                    goal_atoms.append(f"{pred_name}({', '.join(obj_names)})")
            
            if goal_atoms:
                return "Achieve: " + ", ".join(goal_atoms)
        
        # Fallback to string representation if atoms extraction fails
        return f"Goal: {str(goal_obj)}"

    def _parse_plan_from_text(self, plan_text: str, obs: _O, info: dict[str, Any]) -> List[_U]:
        """Parse the VLM output into a list of actions."""
        lines = plan_text.split('\n')
        plan_actions = []
        
        # Find the "Plan:" section
        plan_start = -1
        for i, line in enumerate(lines):
            if line.strip().lower().startswith('plan:'):
                plan_start = i + 1
                break
        
        if plan_start == -1:
            raise VLMPlanningAgentFailure("Could not find 'Plan:' section in VLM output")
        
        # Parse each plan step
        for line in lines[plan_start:]:
            line = line.strip()
            if not line or not any(c.isalnum() for c in line):
                continue
                
            # Remove numbering (e.g., "1. ", "2. ", etc.)
            import re
            line = re.sub(r'^\d+\.\s*', '', line)
            
            # Try to parse action arrays from the text
            if line:
                action = self._parse_action_from_text(line)
                if action is not None:
                    plan_actions.append(cast(_U, action))
                
                if len(plan_actions) >= self._max_planning_horizon:
                    break
        
        if not plan_actions:
            raise VLMPlanningAgentFailure("No valid actions parsed from VLM output")
            
        return plan_actions

    def _parse_action_from_text(self, action_text: str) -> Optional[np.ndarray]:
        """Parse a single action from text."""
        import re
        
        # Look for array-like patterns [a, b, c, d, e]
        array_match = re.search(r'\[([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?(?:\s*,\s*[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)*)\]', action_text)
        if array_match:
            try:
                # Parse the numbers from the array
                numbers_str = array_match.group(1)
                numbers = [float(x.strip()) for x in numbers_str.split(',')]
                return np.array(numbers, dtype=np.float32)
            except (ValueError, TypeError):
                pass
        
        # No valid action found
        raise ValueError(f"Unable to parse action from text: {action_text}")

    def update(self, obs: _O, reward: float, done: bool, info: dict[str, Any]) -> None:
        """Update the agent with new observation."""
        # For now, we don't replan during execution
        # Could be extended to replan if needed
        pass