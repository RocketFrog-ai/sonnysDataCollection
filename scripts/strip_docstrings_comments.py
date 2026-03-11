import ast
import sys
from pathlib import Path

class DocstringRemover(ast.NodeTransformer):

    def _remove_docstring(self, node):
        if ast.get_docstring(node):
            node.body = node.body[1:] if node.body else []
        self.generic_visit(node)
        return node

    def visit_Module(self, node):
        if ast.get_docstring(node):
            node.body = node.body[1:] if node.body else []
        self.generic_visit(node)
        return node

    def visit_FunctionDef(self, node):
        return self._remove_docstring(node)

    def visit_AsyncFunctionDef(self, node):
        return self._remove_docstring(node)

    def visit_ClassDef(self, node):
        return self._remove_docstring(node)

def strip_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            source = f.read()
    except Exception as e:
        print(f'  Skip {filepath}: {e}', file=sys.stderr)
        return False
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        print(f'  Skip {filepath} (syntax): {e}', file=sys.stderr)
        return False
    DocstringRemover().visit(tree)
    try:
        new_source = ast.unparse(tree)
    except Exception as e:
        print(f'  Skip {filepath} (unparse): {e}', file=sys.stderr)
        return False
    result = new_source + ('\n' if not new_source.endswith('\n') else '')
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(result)
    return True

def main():
    root = Path(__file__).resolve().parent.parent
    exclude = {'.git', '__pycache__', 'venv', '.venv', 'node_modules'}
    count = 0
    for py in root.rglob('*.py'):
        if any((x in py.parts for x in exclude)):
            continue
        if strip_file(py):
            count += 1
            print(py.relative_to(root))
    print(f'\nProcessed {count} files')
if __name__ == '__main__':
    main()
