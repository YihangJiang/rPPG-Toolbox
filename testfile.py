# %%
import resource
# %%

print(f"Memory usage: {(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024):.2f} MB")
# %%
