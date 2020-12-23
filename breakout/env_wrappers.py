
import tf_agents.environments.wrappers

def list_tfagent_wrappers():
    for name in dir(tf_agents.environments.wrappers):
        obj = getattr(tf_agents.environments.wrappers, name)
        if hasattr(obj, "__base__") and issubclass(obj, tf_agents.environments.wrappers.PyEnvironmentBaseWrapper):
            print("{:27s} {}".format(name, obj.__doc__.split("\n")[0]))