DEFAULT_FUNCTION_JINJA_TEMPLATE = """You are a world class programmer that writes the highest quality code in terms of readability and efficiency. You reply concisely by only providing what is asked of you. Wrap the code you write in <START-OF-CODE> and <END-OF-CODE> tags and make sure to provide a docstring. Also make sure to import any needed packages for the code to run by itself.
You are asked to write a {{ object_type }} in Python {% if input_types != None %} that takes in {{ input_types|length }} arguments with types {{input_types}} {% if output_types != None %} and outputs {{output_types}}{% endif %}{% endif %} following the following prompt provided by the user: "{{prompt}}"
{% if unit_tests != None %}
The following are the unit tests that the {{ object_type }} has to satisfy:
{% for unit_test in unit_tests %}
Input: {{unit_test[0]}}
Output: {{unit_test[1]}}
{% endfor %}
{% endif %}
"""
