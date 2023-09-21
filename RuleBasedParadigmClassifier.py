import ast

# if else elif keywords -> Imperative
class IfElseElifVisitor(ast.NodeVisitor):
    def __init__(self):
        self.has_if_else_elif = False
    def visit_If(self, node):
        self.has_if_else_elif = True
        self.generic_visit(node)
    def visit_IfExp(self, node): # This will account for inline if-else expressions
        self.has_if_else_elif = True
        self.generic_visit(node)
def contains_if_else_elif(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except Exception as e:
        # 문법 오류가 있는 코드는 분석할 수 없습니다.
        print(f"Error parsing code: {e}")
        return False
    visitor = IfElseElifVisitor()
    visitor.visit(tree)
    return visitor.has_if_else_elif

# while keywords -> Imperative
class WhileVisitor(ast.NodeVisitor):
    def __init__(self):
        self.has_while = False
    def visit_While(self, node):
        self.has_while = True
        self.generic_visit(node)
def contains_while(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except Exception as e:
        # 문법 오류가 있는 코드는 분석할 수 없습니다.
        print(f"Error Parsing Code: {e}")
        return False
    visitor = WhileVisitor()
    visitor.visit(tree)
    return visitor.has_while

# break keywords -> Imperative
class BreakVisitor(ast.NodeVisitor):
    def __init__(self):
        self.has_break = False
    def visit_Break(self, node):
        self.has_break = True
def contains_break(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except Exception as e:
        # 문법 오류가 있는 코드는 분석할 수 없습니다.
        print(f"Error Parsing Code: {e}")
        return False
    visitor = BreakVisitor()
    visitor.visit(tree)
    return visitor.has_break

# continue keywords -> Imperative
class ContinueVisitor(ast.NodeVisitor):
    def __init__(self):
        self.has_continue = False
    def visit_Continue(self, node):
        self.has_continue = True
def contains_continue(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except Exception as e:
        # 문법 오류가 있는 코드는 분석할 수 없습니다.
        print(f"Error Parsing Code: {e}")
        return False
    visitor = ContinueVisitor()
    visitor.visit(tree)
    return visitor.has_continue

# assert keywords -> Imperative
class AssertVisitor(ast.NodeVisitor):
    def __init__(self):
        self.has_assert = False
    def visit_Assert(self, node):
        self.has_assert = True
def contains_assert(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except Exception as e:
        # 문법 오류가 있는 코드는 분석할 수 없습니다.
        print(f"Error Parsing Code: {e}")
        return False
    visitor = AssertVisitor()
    visitor.visit(tree)
    return visitor.has_assert

# del keywords -> Imperative
class DelVisitor(ast.NodeVisitor):
    def __init__(self):
        self.has_del = False
    def visit_Delete(self, node):
        self.has_del = True
def contains_del(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except Exception as e:
        # 문법 오류가 있는 코드는 분석할 수 없습니다.
        print(f"Error Parsing Code: {e}")
        return False
    visitor = DelVisitor()
    visitor.visit(tree)
    return visitor.has_del

# array indexing -> Imperative
    # code syntax error 있으면, 못씀
    # code1 = "def _findNearest(arr, value):\n    \"\"\" Finds the value in arr that value is closest to\n    \"\"\"\n    arr = np.array(arr)\n    # find nearest value in array\n    idx = (abs(arr-value)).argmin()\n    return arr[idx]"
    # print(contains_array_indexing(code1))
class ArrayIndexingVisitor(ast.NodeVisitor):
    def __init__(self):
        self.has_indexing = False
    def visit_Subscript(self, node):
        if isinstance(node.slice, ast.Index):
            self.has_indexing = True
        self.generic_visit(node)
def contains_array_indexing(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except Exception as e:
        # 문법 오류가 있는 코드는 분석할 수 없습니다.
        print(f"Error Parsing Code: {e}")
        return False
    visitor = ArrayIndexingVisitor()
    visitor.visit(tree)
    return visitor.has_indexing

# pass keywords -> Imperative
class PassVisitor(ast.NodeVisitor):
    def __init__(self):
        self.has_pass = False
    def visit_Pass(self, node):
        self.has_pass = True
def contains_pass(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except Exception as e:
        # 문법 오류가 있는 코드는 분석할 수 없습니다.
        print(f"Error Parsing Code: {e}")
        return False
    visitor = PassVisitor()
    visitor.visit(tree)
    return visitor.has_pass

# return keywords -> Imperative
class ReturnVisitor(ast.NodeVisitor):
    def __init__(self):
        self.has_return = False
    def visit_Return(self, node):
        self.has_return = True
def contains_return(code_str):
    try:
        tree = ast.parse(code_str)
    except Exception as e:
        # 문법 오류가 있는 코드는 분석할 수 없습니다.
        print(f"Error Parsing Code: {e}")
        return False
    visitor = ReturnVisitor()
    visitor.visit(tree)
    return visitor.has_return

# def keywords -> Imperative
class FunctionDefFinder(ast.NodeVisitor):
    def __init__(self):
        self.has_def = False
    def visit_FunctionDef(self, node):
        self.has_def = True
def contains_def(code_str):
    try:
        tree = ast.parse(code_str)
    except Exception as e:
        # 문법 오류가 있는 코드는 분석할 수 없습니다.
        print(f"Error Parsing Code: {e}")
        return False
    visitor = FunctionDefFinder()
    visitor.visit(tree)
    return visitor.has_def

# with keywords -> Imperative
class WithFinder(ast.NodeVisitor):
    def __init__(self):
        self.has_with = False
    def visit_With(self, node):
        self.has_with = True
def contains_with(code_str): 
    try:
        tree = ast.parse(code_str)
    except Exception as e:
        # 문법 오류가 있는 코드는 분석할 수 없습니다.
        print(f"Error Parsing Code: {e}")
        return False
    finder = WithFinder()
    finder.visit(tree)
    return finder.has_with

# try keywords -> Imperative
class TryVisitor(ast.NodeVisitor):
    def __init__(self):
        self.has_try = False   
    def visit_Try(self, node):
        self.has_try = True
def contains_try(code_str):
    try:
        tree = ast.parse(code_str)
    except Exception as e:
        # 문법 오류가 있는 코드는 분석할 수 없습니다.
        print(f"Error Parsing Code: {e}")
        return False
    visitor = TryVisitor()
    visitor.visit(tree)
    return visitor.has_try

# except keywords -> Imperative
class ExceptVisitor(ast.NodeVisitor):
    def __init__(self):
        self.has_except = False
    def visit_ExceptHandler(self, node):
        self.has_except = True
def contains_except(python_code):
    try:
        tree = ast.parse(python_code)
    except Exception as e:
        # 문법 오류가 있는 코드는 분석할 수 없습니다.
        print(f"Error Parsing Code: {e}")
        return False
    visitor = ExceptVisitor()
    visitor.visit(tree)
    return visitor.has_except

# finally keywords -> Imperative
class FinallyVisitor(ast.NodeVisitor):
    def __init__(self):
        self.has_finally = False
    def visit_Try(self, node):
        if node.finalbody:  # 이 노드는 'finally' 블록을 포함합니다.
            self.has_finally = True
        self.generic_visit(node)  # 다른 하위 노드를 계속 방문합니다.
def contains_finally(code_string):
    try:
        tree = ast.parse(code_string)
    except Exception as e:
        # 문법 오류가 있는 코드는 분석할 수 없습니다.
        print(f"Error Parsing Code: {e}")
        return False
    visitor = FinallyVisitor()
    visitor.visit(tree)
    return visitor.has_finally

# raise keywords -> Imperative
class RaiseVisitor(ast.NodeVisitor):
    def __init__(self):
        self.has_raise = False
    def visit_Raise(self, node):
        self.has_raise = True
def contains_raise(code_string):
    try:
        tree = ast.parse(code_string)
    except Exception as e:
        # 문법 오류가 있는 코드는 분석할 수 없습니다.
        print(f"Error Parsing Code: {e}")
        return False
    visitor = RaiseVisitor()
    visitor.visit(tree)
    return visitor.has_raise

#######################################################################

# function as arg -> functional
    # code syntax error 있으면, 못씀
    # code1 = "def get_file_size(filename):\n    \"\"\"\n    Get the file size of a given file\n\n    :param filename: string: pathname of a file\n    :return: human readable filesize\n    \"\"\"\n    if os.path.isfile(filename):\n        return convert_size(os.path.getsize(filename))\n    return None"
    # print(contains_function_as_arg(code1))
class FunctionAsArgVisitor(ast.NodeVisitor):
    def __init__(self):
        self.has_function_as_arg = False
        self.defined_functions = set()
    def visit_FunctionDef(self, node):
        self.defined_functions.add(node.name)
        self.generic_visit(node)
    def visit_Call(self, node):
        for arg in node.args:
            # 함수 호출을 인자로 가지는 경우 확인
            if isinstance(arg, ast.Call):
                self.has_function_as_arg = True
                break
            # 함수 참조를 인자로 가지는 경우 확인
            if isinstance(arg, ast.Name) and arg.id in self.defined_functions:
                self.has_function_as_arg = True
                break
        self.generic_visit(node)
def contains_function_as_arg(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except Exception as e:
        # 문법 오류가 있는 코드는 분석할 수 없습니다. 
        print(f"Error Parsing Code: {e}")
        return False
    visitor = FunctionAsArgVisitor()
    visitor.visit(tree)
    return visitor.has_function_as_arg

# lambda functions -> functional
    # code syntax error 있으면, 못씀
    # code1 = "def format_result(input):\n        \"\"\"From: http://stackoverflow.com/questions/13062300/convert-a-dict-to-sorted-dict-in-python\n        \"\"\"\n        items = list(iteritems(input))\n        return OrderedDict(sorted(items, key=lambda x: x[0]))"
    # print(contains_lambda_function(code1))
class LambdaVisitor(ast.NodeVisitor):
    def __init__(self):
        self.has_lambda = False
    def visit_Lambda(self, node):
        self.has_lambda = True
def contains_lambda_function(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except Exception as e:
        # 문법 오류가 있는 코드는 분석할 수 없습니다.
        print(f"Error Parsing Code: {e}")
        return False
    visitor = LambdaVisitor()
    visitor.visit(tree)
    return visitor.has_lambda

# list comprehension -> functional
    # code syntax error 있으면, 못씀
    # code1 = "def uniq(seq):\n    \"\"\" Return a copy of seq without duplicates. \"\"\"\n    seen = set()\n    return [x for x in seq if str(x) not in seen and not seen.add(str(x))]"
    # print(contains_list_comprehension(code1))
class ListCompVisitor(ast.NodeVisitor):
    def __init__(self):
        self.has_list_comp = False
    def visit_ListComp(self, node):
        self.has_list_comp = True
def contains_list_comprehension(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except Exception as e:
        # 문법 오류가 있는 코드는 분석할 수 없습니다.
        print(f"Error Parsing Code: {e}")
        return False
    visitor = ListCompVisitor()
    visitor.visit(tree)
    return visitor.has_list_comp

# decorators -> functional
    # code syntax error 있으면, 못씀
    # code1 = "def install():\n        \"\"\"\n        Installs ScoutApm SQL Instrumentation by monkeypatching the `cursor`\n        method of BaseDatabaseWrapper, to return a wrapper that instruments any\n        calls going through it.\n        \"\"\"\n\n        @monkeypatch_method(BaseDatabaseWrapper)\n        def cursor(original, self, *args, **kwargs):\n            result = original(*args, **kwargs)\n            return _DetailedTracingCursorWrapper(result, self)\n\n        logger.debug(\"Monkey patched SQL\")"
    # print(contains_decorator(code1))
class DecoratorVisitor(ast.NodeVisitor):
    def __init__(self):
        self.has_decorator = False
    def visit_FunctionDef(self, node):
        if node.decorator_list:
            self.has_decorator = True
        self.generic_visit(node)
def contains_decorator(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except Exception as e:
        # 문법 오류가 있는 코드는 분석할 수 없습니다. 
        print(f"Error Parsing Code: {e}")
        return False
    visitor = DecoratorVisitor()
    visitor.visit(tree)
    return visitor.has_decorator

# generator expressions -> functional
    # code syntax error 있으면, 못씀
    # code1 = """\ngenerator = (num ** 2 for num in range(10))\nfor num in generator:\n    print(num)\n"""
    # print(contains_generator_expression(code1))
class GeneratorExpVisitor(ast.NodeVisitor):
    def __init__(self):
        self.has_generator_exp = False
    def visit_GeneratorExp(self, node):
        self.has_generator_exp = True
def contains_generator_expression(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except Exception as e:
        # 문법 오류가 있는 코드는 분석할 수 없습니다. 
        print(f"Error Parsing Code: {e}")
        return False
    visitor = GeneratorExpVisitor()
    visitor.visit(tree)
    return visitor.has_generator_exp

# send into generator -> functional
    # code syntax error 있으면, 못씀
    # code1 = """\ngen = (x for x in range(10))\nvalue = gen.send(None)\n"""
    # print(contains_send_function(code1))
class SendMethodVisitor(ast.NodeVisitor):
    def __init__(self):
        self.has_send_method = False
    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute) and node.func.attr == "send":
            self.has_send_method = True
        self.generic_visit(node)
def contains_send_function(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except Exception as e:
        # 문법 오류가 있는 코드는 분석할 수 없습니다. 
        print(f"Error Parsing Code: {e}")
        return False
    visitor = SendMethodVisitor()
    visitor.visit(tree)
    return visitor.has_send_method

# iter -> functional
    # code syntax error 있으면, 못씀
    # code1 = "def itervalues(d, **kw):\n    \"\"\"Return an iterator over the values of a dictionary.\"\"\"\n    if not PY2:\n        return iter(d.values(**kw))\n    return d.itervalues(**kw)"
    # print(contains_iter_function(code1))
class IterFunctionVisitor(ast.NodeVisitor):
    def __init__(self):
        self.has_iter_function = False
    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id == "iter":
            self.has_iter_function = True
        self.generic_visit(node)
def contains_iter_function(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except Exception as e:
        # 문법 오류가 있는 코드는 분석할 수 없습니다. 
        print(f"Error Parsing Code: {e}")
        return False
    visitor = IterFunctionVisitor()
    visitor.visit(tree)
    return visitor.has_iter_function

# map -> functional
    # code syntax error 있으면, 못씀
    # code1 = "def NeuralNetLearner(dataset, sizes):\n   \"\"\"Layered feed-forward network.\"\"\"\n\n   activations = map(lambda n: [0.0 for i in range(n)], sizes)\n   weights = []\n\n   def predict(example):\n      unimplemented()\n\n   return predict"
    # print(contains_map_function(code1))
class MapFunctionVisitor(ast.NodeVisitor):
    def __init__(self):
        self.has_map_function = False
    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id == "map":
            self.has_map_function = True
        self.generic_visit(node)
def contains_map_function(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except Exception as e:
        # 문법 오류가 있는 코드는 분석할 수 없습니다. 
        print(f"Error Parsing Code: {e}")
        return False
    visitor = MapFunctionVisitor()
    visitor.visit(tree)
    return visitor.has_map_function

# sorted -> functional
    # code syntax error 있으면, 못씀
    # code1 = "def _dict_values_sorted_by_key(dictionary):\n    # This should be a yield from instead.\n    \"\"\"Internal helper to return the values of a dictionary, sorted by key.\n    \"\"\"\n    for _, value in sorted(dictionary.iteritems(), key=operator.itemgetter(0)):\n        yield value"
    # print(contains_sorted_function(code1))
class SortedFunctionVisitor(ast.NodeVisitor):
    def __init__(self):
        self.has_sorted_function = False
    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id == "sorted":
            self.has_sorted_function = True
        self.generic_visit(node)
def contains_sorted_function(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except Exception as e:
        # 문법 오류가 있는 코드는 분석할 수 없습니다. 
        print(f"Error Parsing Code: {e}")
        return False
    visitor = SortedFunctionVisitor()
    visitor.visit(tree)
    return visitor.has_sorted_function

# filter -> functional
    # code syntax error 있으면, 못씀
    # code1 = "def unpunctuate(s, *, char_blacklist=string.punctuation):\n    \"\"\" Remove punctuation from string s. \"\"\"\n    # remove punctuation\n    s = \"\".join(c for c in s if c not in char_blacklist)\n    # remove consecutive spaces\n    return \" \".join(filter(None, s.split(\" \")))"
    # print(contains_filter_function(code1))
class FilterFunctionVisitor(ast.NodeVisitor):
    def __init__(self):
        self.has_filter_function = False
    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id == "filter":
            self.has_filter_function = True
        self.generic_visit(node)
def contains_filter_function(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except Exception as e:
        # 문법 오류가 있는 코드는 분석할 수 없습니다. 
        print(f"Error Parsing Code: {e}")
        return False
    visitor = FilterFunctionVisitor()
    visitor.visit(tree)
    return visitor.has_filter_function

# any -> functional
    # code syntax error 있으면, 못씀
    # code1 = "\ncur = 3\ntemp = [1,3,6,2]\nif any(cur<num for num in temp):\n	print(\"There exist number that is larger than 3\")\n"
    # print(contains_any_function(code1))
class AnyFunctionVisitor(ast.NodeVisitor):
    def __init__(self):
        self.has_any_function = False
    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id == "any":
            self.has_any_function = True
        self.generic_visit(node)
def contains_any_function(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except Exception as e:
        # 문법 오류가 있는 코드는 분석할 수 없습니다. 
        print(f"Error Parsing Code: {e}")
        return False
    visitor = AnyFunctionVisitor()
    visitor.visit(tree)
    return visitor.has_any_function

# all -> functional
    # code syntax error 있으면, 못씀
    # code1 = "def is_nullable_list(val, vtype):\n    \"\"\"Return True if list contains either values of type `vtype` or None.\"\"\"\n    return (isinstance(val, list) and\n            any(isinstance(v, vtype) for v in val) and\n            all((isinstance(v, vtype) or v is None) for v in val))"
    # print(contains_all_function(code1))
class AllFunctionVisitor(ast.NodeVisitor):
    def __init__(self):
        self.has_all_function = False
    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id == "all":
            self.has_all_function = True
        self.generic_visit(node)
def contains_all_function(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except Exception as e:
        # 문법 오류가 있는 코드는 분석할 수 없습니다. 
        print(f"Error Parsing Code: {e}")
        return False
    visitor = AllFunctionVisitor()
    visitor.visit(tree)
    return visitor.has_all_function

# itertools -> functional
    # code syntax error 있으면, 못씀
    # code1 = "def _indexes(arr):\n    \"\"\" Returns the list of all indexes of the given array.\n\n    Currently works for one and two-dimensional arrays\n\n    \"\"\"\n    myarr = np.array(arr)\n    if myarr.ndim == 1:\n        return list(range(len(myarr)))\n    elif myarr.ndim == 2:\n        return tuple(itertools.product(list(range(arr.shape[0])),\n                                       list(range(arr.shape[1]))))\n    else:\n        raise NotImplementedError('Only supporting arrays of dimension 1 and 2 as yet.')"
    # print(contains_itertools_function(code1))
class ItertoolsFunctionVisitor(ast.NodeVisitor):
    def __init__(self):
        self.has_itertools_function = False
    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "itertools":
            self.has_itertools_function = True
        self.generic_visit(node)
def contains_itertools_function(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except Exception as e:
        # 문법 오류가 있는 코드는 분석할 수 없습니다.
        print(f"Error Parsing Code: {e}")
        return False
    visitor = ItertoolsFunctionVisitor()
    visitor.visit(tree)
    return visitor.has_itertools_function

# functools -> functional
    # code syntax error 있으면, 못씀
    # code1 = "def compose_all(tups):\n  \"\"\"Compose all given tuples together.\"\"\"\n  from . import ast  # I weep for humanity\n  return functools.reduce(lambda x, y: x.compose(y), map(ast.make_tuple, tups), ast.make_tuple({}))"
    # print(contains_functools_function(code1))
class FunctoolsFunctionVisitor(ast.NodeVisitor):
    def __init__(self):
        self.has_functools_function = False
    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "functools":
            self.has_functools_function = True
        self.generic_visit(node)
def contains_functools_function(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except Exception as e:
        # 문법 오류가 있는 코드는 분석할수 없습니다.
        print(f"Error Parsing Code: {e}")
        return False
    visitor = FunctoolsFunctionVisitor()
    visitor.visit(tree)
    return visitor.has_functools_function

# enumerate -> functional
    # code syntax error 있으면, 못씀
    # code1 = "def get_lines(handle, line):\n    \"\"\"\n    Get zero-indexed line from an open file-like.\n    \"\"\"\n    for i, l in enumerate(handle):\n        if i == line:\n            return l"
    # print(contains_enumerate_function(code1))
class EnumerateFunctionVisitor(ast.NodeVisitor):
    def __init__(self):
        self.has_enumerate_function = False
    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id == "enumerate":
            self.has_enumerate_function = True
        self.generic_visit(node)
def contains_enumerate_function(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except Exception as e:
        # 문법 오류가 없는 코드는 분석할 수 없습니다.
        print(f"Error Parsing Code: {e}")
        return False
    visitor = EnumerateFunctionVisitor()
    visitor.visit(tree)
    return visitor.has_enumerate_function

# zip -> functional
    # code syntax error 있으면, 못씀
    # code1 = "def column_stack_2d(data):\n    \"\"\"Perform column-stacking on a list of 2d data blocks.\"\"\"\n    return list(list(itt.chain.from_iterable(_)) for _ in zip(*data))"
    # print(contains_zip_function(code1))
class ZipFunctionVisitor(ast.NodeVisitor):
    def __init__(self):
        self.has_zip_function = False
    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id == "zip":
            self.has_zip_function = True
        self.generic_visit(node)
def contains_zip_function(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except Exception as e:
        # 문법 오류가 없는 코드는 분석할 수 없습니다.
        print(f"Error Parsing Code: {e}")
        return False
    visitor = ZipFunctionVisitor()
    visitor.visit(tree)
    return visitor.has_zip_function

# method call count -> functional
class MethodCallCounter(ast.NodeVisitor):
    def __init__(self):
        self.count = 0
    def visit_Attribute(self, node):
        if isinstance(node.ctx, ast.Load):
            self.count += 1
        self.generic_visit(node)
def count_method_calls(code):
    try:
        tree = ast.parse(code)
    except Exception as e:
        # 문법 오류가 없는 코드는 분석할 수 없습니다.
        print(f"Error Parsing Code: {e}")
        return 0
    counter = MethodCallCounter()
    counter.visit(tree)
    return counter.count



def codeToken2codeStr(tokens):

    code = ""
    indentation_level = 0

    for token in tokens:
        if token == "NEW_LINE":
            code += "\n"
            code += "    " * indentation_level
        elif token == "INDENT":
            indentation_level += 1
            code += "    "
        elif token == "DEDENT":
            indentation_level -= 1
            code = code[:-4]
        else:
            code += token + " "

    codeStr = code.strip()

    return codeStr







class ParadigmFeature(object):
    def __init__(self,
                 if_else_elif,
                 while_,
                 break_,
                 continue_,
                 assert_,
                 del_,
                 array_indexing,
                 pass_,
                 return_,
                 def_,
                 with_,
                 try_,
                 except_,
                 finally_,
                 raise_,
                 function_as_arg,
                 lambda_function,
                 list_comprehension,
                 decorator,
                 generator_expression,
                 send_function,
                 iter_function,
                 map_function,
                 sorted_function,
                 filter_function,
                 any_function,
                 all_function,
                 itertools_function,
                 functools_function,
                 enumerate_function,
                 zip_function,
                 cnt_method_calls):
        self.if_else_elif = if_else_elif
        self.while_ = while_
        self.break_ = break_
        self.continue_ = continue_
        self.assert_ = assert_
        self.del_ = del_
        self.array_indexing = array_indexing
        self.pass_ = pass_
        self.return_ = return_
        self.def_ = def_
        self.with_ = with_
        self.try_ = try_
        self.except_ = except_
        self.finally_ = finally_
        self.raise_ = raise_
        self.function_as_arg = function_as_arg
        self.lambda_function = lambda_function
        self.list_comprehension = list_comprehension
        self.decorator = decorator
        self.generator_expression = generator_expression
        self.send_function = send_function
        self.iter_function = iter_function
        self.map_function = map_function
        self.sorted_function = sorted_function
        self.filter_function = filter_function
        self.any_function = any_function
        self.all_function = all_function
        self.itertools_function = itertools_function
        self.functools_function = functools_function
        self.enumerate_function = enumerate_function
        self.zip_function = zip_function
        self.cnt_method_calls = cnt_method_calls

# Paradigm Feature Checker
def FeatureChecker(input_code):
    
    # Imerative Feature Check
    if_else_elif = contains_if_else_elif(input_code)
    while_ = contains_while(input_code)
    break_ = contains_break(input_code)
    continue_ = contains_continue(input_code)
    assert_ = contains_assert(input_code)
    del_ = contains_del(input_code)
    array_indexing = contains_array_indexing(input_code)
    pass_ = contains_pass(input_code)
    return_ = contains_return(input_code)
    def_ = contains_def(input_code)
    with_ = contains_with(input_code)
    try_ = contains_try(input_code)
    except_ = contains_except(input_code)
    finally_ = contains_finally(input_code)
    raise_ = contains_raise(input_code)

    # Functional Feature Check
    function_as_arg = contains_function_as_arg(input_code)
    lambda_function = contains_lambda_function(input_code)
    list_comprehension = contains_list_comprehension(input_code)
    decorator = contains_decorator(input_code)
    generator_expression = contains_generator_expression(input_code)
    send_function = contains_send_function(input_code)
    iter_function = contains_iter_function(input_code)
    map_function = contains_map_function(input_code)
    sorted_function = contains_sorted_function(input_code)
    filter_function = contains_filter_function(input_code)
    any_function = contains_any_function(input_code)
    all_function = contains_all_function(input_code)
    itertools_function = contains_itertools_function(input_code)
    functools_function = contains_functools_function(input_code)
    enumerate_function = contains_enumerate_function(input_code)
    zip_function = contains_zip_function(input_code)
    cnt_method_calls = count_method_calls(input_code)

    return ParadigmFeature(if_else_elif,
                            while_,
                            break_,
                            continue_,
                            assert_,
                            del_,
                            array_indexing,
                            pass_,
                            return_,
                            def_,
                            with_,
                            try_,
                            except_,
                            finally_,
                            raise_,
                            function_as_arg,
                            lambda_function,
                            list_comprehension,
                            decorator,
                            generator_expression,
                            send_function,
                            iter_function,
                            map_function,
                            sorted_function,
                            filter_function,
                            any_function,
                            all_function,
                            itertools_function,
                            functools_function,
                            enumerate_function,
                            zip_function,
                            cnt_method_calls)

def RuleBasedParadigmClassifier(input_code):

    
    # input_code 를 str 형태로 변환
    if type(input_code) == str:
        pass

    elif type(input_code) == list:
        input_code = codeToken2codeStr(input_code)
    
    else:
        print("Input type not supported")
        return None
    
    # Feature Checker
    feature = FeatureChecker(input_code)



    # # Feature Print
    # features_dict = {
    #     "if_else_elif": feature.if_else_elif,
    #     "while_": feature.while_,
    #     "break_": feature.break_,
    #     "continue_": feature.continue_,
    #     "assert_": feature.assert_,
    #     "del_": feature.del_,
    #     "array_indexing": feature.array_indexing,
    #     "pass_": feature.pass_,
    #     "with_": feature.with_,
    #     "try_": feature.try_,
    #     "except_": feature.except_,
    #     "finally_": feature.finally_,
    #     "raise_": feature.raise_,

    #     "function_as_arg": feature.function_as_arg,
    #     "lambda_function": feature.lambda_function,
    #     "list_comprehension": feature.list_comprehension,
    #     "decorator": feature.decorator,
    #     "generator_expression": feature.generator_expression,
    #     "send_function": feature.send_function,
    #     "iter_function": feature.iter_function,
    #     "map_function": feature.map_function,
    #     "sorted_function": feature.sorted_function,
    #     "filter_function": feature.filter_function,
    #     "any_function": feature.any_function,
    #     "all_function": feature.all_function,
    #     "itertools_function": feature.itertools_function,
    #     "functools_function": feature.functools_function,
    #     "enumerate_function": feature.enumerate_function,
    #     "zip_function": feature.zip_function,

    # }
    # max_length = max(len(key) for key in features_dict.keys())
    # for key, value in features_dict.items():
    #     print("{:<{width}} : {}".format(key, value if type(value) == int else ("O" if value == True else ""), width=max_length))




    # Rule Based Paradigm Classifier
    """
    [Paradigm Feature 별 점수]

    feature.if_else_elif            -1점
    feature.while_                  -1점
    feature.break_                  -1점
    feature.continue_               -1점
    feature.assert_                 -1점
    feature.del_                    -1점
    feature.array_indexing          -1점
    feature.pass_                   -1점
    feature.with_                   -1점
    feature.try_                    -1점
    feature.except_                 -1점
    feature.finally_                -1점
    feature.raise_                  -1점

    feature.function_as_arg         ->functional
    feature.lambda_function         ->functional
    feature.list_comprehension      ->functional
    feature.decorator               ->functional
    feature.generator_expression    ->functional
    feature.send_function           ->functional
    feature.iter_function           ->functional
    feature.map_function            ->functional
    feature.sorted_function         ->functional
    feature.filter_function         ->functional
    feature.any_function            ->functional
    feature.all_function            ->functional
    feature.itertools_function      ->functional
    feature.functools_function      ->functional
    feature.enumerate_function      ->functional
    feature.zip_function            ->functional

    

    [Paradigm 판정 기준]
    functional feature 하나라도 있으면, functional paradigm
    하나도 없는 경우,
    score < 0 : Imperative
    score = 0 : Hybrid
    score > 0 : Functional
    """


    # 점수 계산
    score = 0
    functional_flag = False
    # Imperative
    if feature.if_else_elif:
        score -= 1
    if feature.while_:
        score -= 1
    if feature.break_:
        score -= 1
    if feature.continue_:
        score -= 1
    if feature.assert_:
        score -= 1
    if feature.del_:
        score -= 1
    if feature.array_indexing:
        score -= 1
    if feature.pass_:
        score -= 1
    if feature.return_:
        pass
    if feature.def_:
        pass
    if feature.with_:
        score -= 1
    if feature.try_:
        score -= 1
    if feature.except_:
        score -= 1
    if feature.finally_:
        score -= 1
    if feature.raise_:
        score -= 1
    # Functional
    if feature.function_as_arg:
        functional_flag = True
    if feature.lambda_function:
        functional_flag = True
    if feature.list_comprehension:
        functional_flag = True
    if feature.decorator:
        functional_flag = True
    if feature.generator_expression:
        functional_flag = True
    if feature.send_function:
        functional_flag = True
    if feature.iter_function:
        functional_flag = True
    if feature.map_function:
        functional_flag = True
    if feature.sorted_function:
        functional_flag = True
    if feature.filter_function:
        functional_flag = True
    if feature.any_function:
        functional_flag = True
    if feature.all_function:
        functional_flag = True
    if feature.itertools_function:
        functional_flag = True
    if feature.functools_function:
        functional_flag = True
    if feature.enumerate_function:
        functional_flag = True
    if feature.zip_function:
        functional_flag = True
    if feature.cnt_method_calls > 0:
        pass

    # # score print
    # print("\nScore: {}".format(score))

    if functional_flag == True:
        return "functional", score

    if score < 0:
        return "imperative", score
    elif score > 0:
        return "functional", score
    else:
        return "hybrid", score
        


    






if __name__ == "__main__":

    code1 = """
import ast

class MethodCallCounter(ast.NodeVisitor):
    def __init__(self):
        self.count = 0
    def visit_Attribute(self, node):
        if isinstance(node.ctx, ast.Load):
            self.count += 1
        self.generic_visit(node)
def count_method_calls(code):
    tree = ast.parse(code)
    counter = MethodCallCounter()
    counter.visit(tree)
    return counter.count
"""

    paradigm_, score = RuleBasedParadigmClassifier(code1)
    print(code1)
    print(paradigm_)

