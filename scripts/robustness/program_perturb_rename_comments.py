
"""
Code for perturbing Python programs by renaming variables and handling comments.
Adapted from program_purturb_cst.py, focusing on variable renaming and comment removal.
Uses random names from WordNet and validates executability.
"""
import ast
import libcst as cst
import astor
import random
from nltk.corpus import wordnet
import re
import keyword
import csv

WORDS = list(wordnet.all_synsets())

def generate_random_identifier(psi=5):
    i = 0
    identifier = []
    while i < psi:
        word = random.choice(WORDS).lemmas()[0].name()
        if (i == 0 and (not word[0].islower() or word in keyword.kwlist)) or re.search(r'[^a-zA-Z0-9]', word):
            continue
        identifier.append(word)
        i += 1
    return '_'.join(identifier)

def remove_comments(og_code):
    class CommentRemover(cst.CSTTransformer):
        def leave_Comment(self, original_node, updated_node):
            return cst.RemoveFromParent()

    module = cst.parse_module(og_code)
    module.visit(CommentRemover())
    return module.code

def t_remove_comments(module, uid=1, psi=5):
    class CommentCollector(cst.CSTVisitor):
        def __init__(self):
            self.comments = 0

        def visit_Comment(self, node):
            self.comments += 1

    class CommentRemover(cst.CSTTransformer):
        def __init__(self, choice):
            self.count = 0
            self.choice = choice

        def leave_Comment(self, original_node, updated_node):
            if self.count == self.choice:
                return cst.RemoveFromParent()
            self.count += 1
            return updated_node

    collector = CommentCollector()
    module.visit(collector)

    if not collector.comments:
        return False, module.code

    choice = random.randrange(0, collector.comments)

    module.visit(CommentRemover(choice))

    return True, module.code

def t_rename_variable_in_iterator(module, uid=1, psi=5):
    class VariableVisitor(cst.CSTVisitor):
        def __init__(self):
            self.iter_vars = []

        def visit_For(self, node):
            try:
                self.iter_vars.append(node.target.value)
            except:
                pass

        def visit_While(self, node):
            if hasattr(node, "target"):
                self.iter_vars.append(node.target.value)

    class VariableRenamer(cst.CSTTransformer):
        def __init__(self, selection):
            self.selection = selection
            self.uid = uid

        def leave_For(self, original_node, updated_node):
            return self._rename_loop_variables(updated_node)

        def leave_While(self, original_node, updated_node):
            return self._rename_loop_variables(updated_node)

        def _rename_loop_variables(self, node):
            updated_node = node
            if hasattr(node, "target") and isinstance(node.target, cst.Name) and node.target.value == self.selection:
                updated_node = node.with_changes(target=cst.Name(generate_random_identifier(psi)))

            return updated_node.visit(VariableReferenceRenamer(self.selection))

    class VariableReferenceRenamer(cst.CSTTransformer):
        def __init__(self, selection):
            self.selection = selection
            self.uid = uid

        def leave_Name(self, original_node, updated_node):
            if updated_node.value == self.selection:
                return updated_node.with_changes(value=generate_random_identifier(psi))
            return updated_node

    visitor = VariableVisitor()
    module.visit(visitor)
    iter_vars = visitor.iter_vars
    if len(iter_vars) == 0:
        return False, module.code
    selection = random.choice(iter_vars)
    transformer = VariableRenamer(selection)
    module = module.visit(transformer)
    return True, module.code

def t_rename_parameters(module, uid=1, psi=5):
    class FunctionParameterCollector(cst.CSTVisitor):
        def __init__(self):
            self.function_parameters = {}
            self.idx = 0

        def visit_FunctionDef(self, node):
            function_name = node.name.value
            parameters = [param.name.value for param in node.params.params if param != 'self']
            self.function_parameters[self.idx] = (function_name, parameters)
            self.idx += 1

    class ParameterNameReplacer(cst.CSTTransformer):
        def __init__(self, selection):
            self.uid = uid
            self.selection = selection
            super().__init__()

        def leave_Param(self, node: cst.Param, updated_node: cst.Name) -> cst.Param:
            if updated_node.name.value == self.selection:
                return updated_node.with_changes(value=generate_random_identifier(psi))
            return updated_node

        def leave_Name(self, node: cst.Name, updated_node: cst.Name) -> cst.CSTNode:
            if updated_node.value == self.selection:
                return updated_node.with_changes(value=generate_random_identifier(psi))

            return updated_node

    visitor = FunctionParameterCollector()

    module.visit(visitor)

    params = visitor.function_parameters[0][1]

    if len(params) == 0:
        return False, module.code

    selection = random.choice(params)
    transformer = ParameterNameReplacer(selection)
    module = module.visit(transformer)
    return True, module.code

def t_rename_local_variables(module, uid=1, psi=5):
    class FunctionParameterCollector(cst.CSTVisitor):
        def __init__(self):
            self.function_parameters = set()

        def visit_FunctionDef(self, node):
            for param in node.params.params:
                self.function_parameters.add(param)

    class VariableNameVisitor(cst.CSTVisitor):
        def __init__(self, function_parameters):
            self.names = set()
            self.function_parameters = function_parameters
            super().__init__()

        def visit_Assign(self, node: cst.Name) -> cst.CSTNode:
            for target in node.targets:
                try:
                    if target.target.value not in self.function_parameters:
                        self.names.add(target.target.value)
                except:
                    continue

    class VariableNameReplacer(cst.CSTTransformer):
        def __init__(self, selection):
            self.uid = uid
            self.selection = selection
            super().__init__()

        def leave_Name(self, node: cst.Name, updated_node: cst.CSTNode) -> cst.CSTNode:
            if updated_node.value == self.selection:
                return updated_node.with_changes(value=generate_random_identifier(psi))

            return updated_node

    param_visitor = FunctionParameterCollector()
    module.visit(param_visitor)

    visitor = VariableNameVisitor(param_visitor.function_parameters)

    module.visit(visitor)

    if len(visitor.names) == 0:
        return False, module.code

    selection = random.choice(list(visitor.names))
    transformer = VariableNameReplacer(selection)
    module = module.visit(transformer)
    return True, module.code

def is_executable(code):
    try:
        exec(code)
        return True
    except Exception as e:
        print(f"Execution failed: {e}")
        return False

# depth = number of transformations to apply sequentially
# samples = number of generated purturbated versions
# psi = complexity of the chosen random words
def perturb_rename_and_comments(og_code, depth=5, samples=1, psi=3):
    transforms = []
    DEPTH = depth
    NUM_SAMPLES = samples

    for s in range(NUM_SAMPLES):
        the_seq = []
        for i in range(DEPTH):
            # Randomly choose between renaming variables or removing comments
            if random.choice([True, False]):
                # Rename variables: choose one of the rename functions
                rename_funcs = [t_rename_parameters, t_rename_variable_in_iterator, t_rename_local_variables]
                the_seq.append(random.choice(rename_funcs))
            else:
                # Remove comments
                the_seq.append(t_remove_comments)

        transforms.append(('depth-{}-sample-{}'.format(DEPTH, s+1), t_seq(the_seq, psi), the_seq))

    results = []
    for t_name, t_func, the_seq in transforms:
        try:
            exec(og_code)
        except Exception as ex:
            import traceback
            traceback.print_exc()
            results.append({'changed': False, 't_name': t_name, 'the_seq': the_seq, 'result': og_code, 'executable': False})
            continue
        try:
            cst.parse_module(og_code)
        except:
            og_code = astor.to_source(ast.parse(og_code))

        changed, result = t_func(og_code)
        executable = is_executable(result)
        results.append({'changed': changed, 't_name': t_name, 'the_seq': the_seq, 'result': result, 'executable': executable})
    return results

class t_seq(object):
    def __init__(self, transforms, psi):
        self.transforms = transforms
        self.psi = psi

    def __call__(self, the_ast):
        did_change = False
        cur_ast = the_ast
        for i, t in enumerate(self.transforms):
            cur_ast = cst.parse_module(cur_ast)
            changed, cur_ast = t(cur_ast, i+1, self.psi)
            if changed:
                did_change = True
        return did_change, cur_ast

if __name__ == "__main__":
    csv_file = '/home/sakib/Documents/promptmark/scripts/robustness/data/csvs/exp_4_data.csv'
    rows = []
    with open(csv_file, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    
    for row in rows:
        og_code = row['watermarked_code']
        res = perturb_rename_and_comments(og_code, depth=1, samples=1)
        if res:
            row['perturbed_code_static'] = res[0]['result']
            row['executable_static'] = res[0]['executable']
        else:
            row['perturbed_code_static'] = og_code
            row['executable_static'] = False

        print(f"Processed PID: {row['pid']}")
    
    with open(csv_file, 'w', newline='') as f:
        fieldnames = ['pid', 'original_code', 'watermarked_code', 'perturbed_code_static', 'executable_static']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print("Perturbation completed and saved to CSV.")