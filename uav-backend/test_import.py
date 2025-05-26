# import sys
# import os

# # Add the `colab` directory to the Python path
# colab_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../colab"))
# print("Resolved colab path:", colab_path)  # Debug: Print the resolved path
# sys.path.append(colab_path)

# # Test the import
# try:
#     from colab.path_planning_GA.GA_complex_PF import GeneticAlgorithm, ComplexEnvironment
#     print("Import successful!")
# except ModuleNotFoundError as e:
#     print("Import failed:", e)