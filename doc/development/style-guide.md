# Python Style Guide for Automated Bug Triaging Project

This document outlines the coding style, type annotations, and documentation standards to be followed when contributing to the Automated Bug Triaging project. Adhering to these guidelines will ensure code consistency, readability, and maintainability across the project.

## Table of Contents
1. [Code Style](#code-style)
2. [Type Annotations](#type-annotations)
3. [Documentation](#documentation)

## 1. Code Style

- **PEP 8 Compliance**: Follow the PEP 8 style guide for Python code. This includes:
  - **Indentation**: Use 4 spaces per indentation level.
  - **Line Length**: Limit all lines to a maximum of 79 characters.
  - **Blank Lines**: Separate top-level function and class definitions with two blank lines. Use single blank lines to separate methods inside a class and logical sections within a function.
  - **Imports**: Place all import statements at the top of the file. Group imports in the following order:
    1. Standard library imports
    2. Related third-party imports
    3. Local application imports
    ```python
    import os
    import sys

    import numpy as np
    import pandas as pd

    from src.utils import helper_function
    ```
  - **Naming Conventions**:
    - **Variable names**: `snake_case`
    - **Function names**: `snake_case`
    - **Class names**: `CamelCase`
    - **Constants**: `UPPER_SNAKE_CASE`
  - **Spacing**:
    - Use spaces around operators (e.g., `x = a + b`)
    - Do not use spaces inside parentheses, brackets, or braces (e.g., `my_list[0]`)

## 2. Type Annotations

- **Type Hints**: Use type hints to specify the expected types of function arguments and return values. This improves code readability and helps with static analysis tools.
  - Function definitions should include type hints for all parameters and return types.
  - Use `Optional` for arguments that could be `None`.
  - Use `List`, `Dict`, `Tuple`, etc., for complex types.
  
  ```python
  from typing import List, Optional

  def fetch_issues(issue_ids: List[int]) -> Optional[List[dict]]:
      # Function implementation here
  ```

- **Type Checking**: Run type checkers like `mypy` to ensure type correctness throughout the codebase.

## 3. Documentation

- **Docstrings**: Every module, class, and function should have a docstring describing its purpose and usage.
  - **Modules**: At the top of each module, include a docstring describing the module's purpose.
  - **Classes**: Include a class-level docstring explaining the class's role and any important details.
  - **Functions/Methods**: Include a function-level docstring describing the function's purpose, parameters, return value, and any exceptions raised.

  **Example Function Docstring**:
  ```python
  def process_issue_data(issue_data: dict) -> dict:
      """
      Process raw issue data and return a dictionary with relevant details.

      Args:
          issue_data (dict): A dictionary containing raw issue data fetched from the GitHub API.

      Returns:
          dict: A dictionary containing processed issue information.

      Raises:
          ValueError: If the input data is not valid.
      """
      # Function implementation here
  ```

- **Commenting**:
  - Use comments sparingly and only when the code isn't self-explanatory.
  - Write comments in complete sentences and place them above the relevant code.
  
  ```python
  # Filter issues with exactly one assignee
  filtered_issues = [issue for issue in issues if len(issue['assignees']) == 1]
  ```

- **README and Other Documentation**:
  - Keep the `README.md` file up-to-date with clear instructions on how to run the project, the purpose of each module, and any other relevant information.
  - Include additional documentation in the `doc` directory where necessary (e.g., explanations of complex algorithms).

## Additional Guidelines

- **Error Handling**: Use exceptions appropriately and document any exceptions raised by your functions.
- **Testing**: Ensure that every function and module is tested adequately in the `tests` directory. Write tests that cover different cases, including edge cases.
- **Code Reviews**: Before merging code into the main branch, ensure that it passes code reviews. Reviewers should check for adherence to this style guide, functionality, and completeness.
