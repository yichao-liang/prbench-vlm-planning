"""Environment-specific controller loading utilities."""

from typing import Dict, Optional, Any
import logging
import inspect
import importlib


def get_controllers_for_environment(env_name: str) -> Optional[Dict[str, Any]]:
    """Automatically load all controllers for a given environment from prbench_models.
    
    Args:
        env_name: Name of the environment (e.g., "Motion2D-p1", "StickButton2D-b3")
        
    Returns:
        Dictionary with controller classes, or None if not available
    """
    
    # Extract base environment type from full name and map to module path
    env_module_map = {
        "Motion2D": "prbench_models.geom2d.envs.motion2d.parameterized_skills",
        "StickButton2D": "prbench_models.geom2d.envs.stickbutton2d.parameterized_skills", 
        "Obstruction2D": "prbench_models.geom2d.envs.obstruction2d.parameterized_skills",
        "ClutteredStorage2D": "prbench_models.geom2d.envs.clutteredstorage2d.parameterized_skills",
        "ClutteredRetrieval2D": "prbench_models.geom2d.envs.clutteredretrieval2d.parameterized_skills",
    }
    
    # Find matching environment type
    env_type = None
    for key in env_module_map:
        if key in env_name:
            env_type = key
            break
    
    if not env_type:
        logging.warning(f"No specific controllers available for environment: {env_name}")
        return None
    
    return _import_all_controllers(env_module_map[env_type], env_type)


def _import_all_controllers(module_path: str, env_type: str) -> Optional[Dict[str, Any]]:
    """Import all controller classes from a given module.
    
    Args:
        module_path: Python import path to the parameterized_skills module
        env_type: Environment type name for logging
        
    Returns:
        Dictionary with all controller classes, or None if import fails
    """
    try:
        # Import the module
        module = importlib.import_module(module_path)
        
        # Import the base controller class to check inheritance
        try:
            from prbench_models.geom2d.utils import Geom2dRobotController
        except ImportError:
            # Fallback to name-based filtering if we can't import the base class
            logging.warning("Could not import Geom2dRobotController, falling back to name-based filtering")
            Geom2dRobotController = None
        
        # Find all controller classes that are subclasses of Geom2dRobotController
        controllers = {}
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if not name.startswith('_'):
                # Check if it's a subclass of Geom2dRobotController
                if Geom2dRobotController and issubclass(obj, Geom2dRobotController) and obj != Geom2dRobotController:
                    controllers[name] = obj
                # Fallback: check if name suggests it's a controller
                elif Geom2dRobotController is None and name.endswith('Controller'):
                    controllers[name] = obj
        
        if controllers:
            logging.info(f"Found {len(controllers)} controllers for {env_type}: {list(controllers.keys())}")
            return {"controllers": controllers}
        else:
            logging.info(f"No controllers found in {module_path}")
            return None
            
    except ImportError as e:
        logging.info(f"{env_type} controllers not available: {e}")
        return None
    except Exception as e:
        logging.error(f"Error loading controllers from {module_path}: {e}")
        return None


