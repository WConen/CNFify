# Sorry, this is the product of porting a hastily assembled ipython notebook 
# to a (first, very simple) streamlit app (it contains a lot "printable" debug stuff, sorry)
# (if you want access to the notebook, let me know)
# It might improve over time, (if life allows for it...)
# Alas, it seems to work, so that's the point for now.

# F = '((¬A ∨ B) → ¬(A ↔ B))'
'''
TODO:
(1) Make it safe against malformed inputs (no complex error messages for now, just no crashing!), 
    empty formula should also be recognized (DONE)
(2) Include a Debug-Switch with at least two level of debugging: 
    (2.1) level 1: show intermediate results after step 0, 1, and 2
    (2.2) level 2: show each transformation step made explizit, as well as the 
        results after step 0, 1, and 2  (differently solved)
(3) Pick the atomic formulas from the text of the formula yourself (DONE)
(4) give a possibility to number a formula, to show it as a tree, to evaluate it (iii: DONE)
(5) Restrict evaluation to formulas with 5 or less atomic formulas (DONE)
'''

## Erweiterung um Evaluation und Nummerierung
class Tree():
  # unchanged
  def __init__(self,content,parent=None,left=None,right=None):
    if content in ['→','↔','∧','∨']:
      self.c_typ,self.c_val = ("2OP",content)
    elif content == '¬':
      self.c_typ,self.c_val = ("1OP",content)
    else:
      self.c_typ,self.c_val = ("ATOM",content)
    self.parent = parent
    if left: left.parent = self
    self.left = left
    if right: right.parent = self
    self.right = right

  # unchanged
  def __str__(self):
    if self.c_typ == 'ATOM': return self.c_val
    if self.c_typ == '1OP': return self.c_val + str(self.left)
    # Typ 2OP:
    return "(" + str(self.left) + self.c_val + str(self.right) + ')'

  def __eq__(self, other):
    return str(self) == str(other)

  # Unsere Semantik
  def eval(self,belegung):
    if self.c_typ == "ATOM":
      return belegung[self.c_val]
    if self.c_typ == "1OP" and self.c_val == '¬':
      return not self.left.eval(belegung)
    else:
      if self.c_val == '∧':
        return self.left.eval(belegung) and self.right.eval(belegung)
      if self.c_val == '∨':
        return self.left.eval(belegung) or self.right.eval(belegung)
      if self.c_val == '→':
        return (not self.left.eval(belegung)) or self.right.eval(belegung)
      if self.c_val == '↔':
        return (self.left.eval(belegung) and self.right.eval(belegung)) \
              or \
               (not self.left.eval(belegung) and not self.right.eval(belegung))

  def nummer(self):
    if self.c_typ == 'ATOM': return 0
    if self.c_typ == '1OP': return 1 + self.left.nummer()
    # Typ 2OP:
    return max(self.left.nummer(),self.right.nummer()) + 1

  def str_nummerieren(self):
    if self.c_typ == 'ATOM': return self.c_val + '/' + str(self.nummer())
    if self.c_typ == '1OP':
      return self.c_val + '/' + str(self.nummer()) + self.left.str_nummerieren()
    # Typ 2OP:
    return "(" + self.left.str_nummerieren() + ' '\
        + self.c_val + '/' + str(self.nummer()) \
        + ' ' + self.right.str_nummerieren() + ')'

""" FTree = Tree('→',
             left=Tree('∨',
                       left=Tree('¬',
                                 left=Tree('A')),
                       right=Tree('B')),
             right=Tree('¬',
                        left=Tree('↔',
                                  left=Tree('A'),
                                  right=Tree('B'))))
print(FTree)
print(FTree.str_nummerieren())
print(FTree.eval({'A':False,'B':False}))
print(FTree.eval({'A':False,'B':True}))
print(FTree.eval({'A':True,'B':False}))
print(FTree.eval({'A':True,'B':True}))

F = '((¬A ∨ B) → ¬(A ↔ B))'
 """


def preprocess(f):
  atoms = set()
  allowed = ['(',')','→','↔','∧','∨','¬','A','B','C','D','E']
  atom_chars = ['A','B','C','D','E']

  # Clean string
  f = f.replace('<->','↔')
  f = f.replace('->','→')
  f = f.replace('-','¬')
  f = f.replace('n','∧')
  f = f.replace('v','∨')
  f = f.replace(' ','')
  f = f.replace('a','A')
  f = f.replace('b','B')
  f = f.replace('c','C')
  f = f.replace('d','D')
  f = f.replace('e','E')

  # Check for unallowed chars, check for unbalanced parens
  open = 0
  
  for c in f:
    if c not in allowed: return (False,'Forbidden characters in input!',[])
    if c in atom_chars: atoms.add(c)
    if c == '(':
      open += 1
    elif c == ')':
      open -= 1
      if open < 0: return (False,'Too many closed ) too early!',[])
  if open != 0: return (False,'Unbalanced ()! '+str(open)+" not closed",[])
  return (True,'('+f+')',sorted(list(atoms)))

def tokenize(formel):
  op_chars = ['→','↔','∧','∨']
  token = []
  ctx = None

  for c in formel:
    if c == ' ': continue
    if c == '(':
      token.append((0,c))
    elif c == ')':
      token.append((1,c))
    elif c == '¬':
      token.append((2,c))
    elif c in op_chars:
      token.append((3,c))
    else:
      token.append((4,c))
  token.append((6,'')) # END-Token
  return token

# token = tokenize(F)
# print(token)

class Tokenlist():
  def __init__(self,token):
    self.token = token[:]
    self.pos = 0

  def __iter__(self):
    self.pos = 0
    return self

  def __next__(self):
    if self.pos+1 > len(self.token):
      return None
    self.pos += 1
    return self.token[self.pos-1]

  def peek(self,horizont=0):
    pos = self.pos+horizont
    if pos >= len(self.token) or pos < 0: return None
    return self.token[pos]

# tl = Tokenlist(token)

# Regeln: (das | bedeutet ODER)
# OP1 = '¬'
# OP2 = '→' | '↔' | '∧' | '∨'
# FORMEL := ATOM | OP1 FORMEL | '(' FORMEL OP2 FORMEL ')'
# Wir können im Moment nur Atome mit einem Zeichen repräsentieren, reicht

token_typen = ['FORMEL_S','FORMEL_E','OP1','OP2','ATOM','FORMEL','END']
knoten_typen = ['OP1','OP2','ATOM','FORMEL']

# F = '((¬A ∨ B) → ¬(A ↔ B))'

def parse_typ(tl,ctx,typ,debug=0):
  if debug: print(ctx,"Looking for",typ,"!")
  if tl.peek()[0] == typ:
    ttyp,tval = tl.__next__()
    return (token_typen[ttyp],ttyp,tval)
  else:
    if debug: print(ctx,"Error while looking for",typ,"!")
    return None

def parse_formels(tl,ctx,debug):
  return parse_typ(tl,ctx,0,debug)

def parse_formele(tl,ctx,debug):
  return parse_typ(tl,ctx,1,debug)

def parse_op1(tl,ctx,debug):
  return parse_typ(tl,ctx,2,debug)

def parse_op2(tl,ctx,debug):
  return parse_typ(tl,ctx,3,debug)

def parse_atom(tl,ctx,debug):
  return parse_typ(tl,ctx,4,debug)

def parse_formel(tl,ctx,debug=0):
  # ( FORMEL ) or ATOM "alone"
  #            or NEG ATOM
  #            or NEG FORMEL
  #            or FORMEL OP2 FORMEL
  #            or ATOM OP2 FORMEL
  if debug: print(ctx,"Looking for FORMEL")
  tnext = tl.peek()
  if debug: print(ctx,"  tnext:",tnext)
  tnext_typ,tnext_val = tnext
  if tnext_typ == 0: # (
    parse_formels(tl,ctx,debug)
    c1 = parse_formel(tl,ctx+" ",debug)
    c2 = None # ( FORMEL )
    if tl.peek()[0] == 3: # ( FORMEL OP2 FORMEL )
      c2 = parse_op2(tl,ctx+"  ",debug)
      c3 = parse_formel(tl,ctx+" ",debug)
    parse_formele(tl,ctx,debug)
    return c1 if not c2 else ("FORMEL_OP2",c1,c2,c3)
  elif tnext_typ == 4: # ATOM
    c1 = parse_atom(tl,ctx+" ",debug)
    c2 = None # ATOM
    if tl.peek()[0] == 3: # ATOM OP2 FORMEL
      c2 = parse_op2(tl,ctx+"  ",debug)
      c3 = parse_formel(tl,ctx+" ",debug)
    return c1 if not c2 else ("FORMEL_OP2",c1,c2,c3)
  elif tnext_typ == 2: # NEG
    c1 = parse_op1(tl,ctx+" ",debug)
    if tl.peek()[0] == 4: # NEG ATOM
      c2 = parse_atom(tl,ctx+" ",debug)
    else:
      c2 = parse_formel(tl,ctx+" ",debug) # NEG FORMEL
    return("FORMEL_NEG",c2)
  else:
    print("ERROR, unknown type")
    return None

# import pprint
# pprint.pprint(result := parse_formel(tl,"",debug=1))

## Problem noch: Extraschicht FORMEL, falls es Extra-Klammern gibt,
## die noch rausprüfen
""" FTree = Tree('→',
             left=Tree('∨',
                       left=Tree('¬',
                                 left=Tree('A')),
                       right=Tree('B')),
             right=Tree('¬',
                        left=Tree('↔',
                                  left=Tree('A'),
                                  right=Tree('B'))))
print(FTree)
 """
# Currently, we have either 2,3 or 4 Elements
def create_tree(formel):
  typ = formel[0]
  if typ == 'ATOM':
    return Tree(formel[2])
  elif typ == 'FORMEL_NEG':
    return Tree('¬',left=create_tree(formel[1]))
  elif typ == 'FORMEL_OP2':
    return Tree(formel[2][2],
                left=create_tree(formel[1]),
                right=create_tree(formel[3])
    )
  else:
    print("UNKNOWN STUFF...ERROR!")

""" # Uses the result from above, should recreate the same tree
# FTree1 = create_tree(result)
# print(FTree1)

# Total process:

Fs = ['(¬(¬A∨B)→¬(¬A↔B))','((¬A∨B)→¬(¬¬A↔B))','¬(¬(¬A∨¬B)→¬(¬A↔¬B))',
      '¬¬¬¬¬(¬(¬A∨¬B)¬→¬(¬A↔¬B))','¬¬¬¬¬(¬(¬A∨¬B)→¬(¬A↔¬B))']

# Not very verbous when it comes to errors..., see the fourth (invalid) example,
# however, it produces a valid formula at least ¬¬¬¬¬¬(¬A∨¬B)
# ... should this worry us somewhat? Hm, ¬(¬(...)) works out in the 5. example

for f in Fs:
  print(f)
  print(res := create_tree(parse_formel(Tokenlist(tokenize(f)),"")))
  print("MATCHES?",f == str(res),"\n")
"""

import copy

## Create KNF from Formula
#F = '((¬A ∨ B) → ¬(A ↔ B))'
#res = create_tree(parse_formel(Tokenlist(tokenize(F)),""))
 

# Helper function for debugging, checks whether the parent links are correct
def check_parents(t:Tree,ancestor=None,debug=0,pre=0):
  if debug: print(pre*"  ","Check ",t.c_typ,t.c_val)
  result = (t.parent == ancestor)
  if not result:
    print("Error in parent of:",t,"\n  -- Parent:",t.parent,"\n  -- Ancest:",ancestor)
  if t.c_typ == "ATOM": return result
  if t.c_typ == "1OP":
    return result and check_parents(t.left,ancestor=t,debug=debug,pre=pre+1)
  if t.c_typ == "2OP":
    return result and check_parents(t.left,ancestor=t,debug=debug,pre=pre+1) \
      and check_parents(t.right,ancestor=t,debug=debug,pre=pre+1)

## Work on the Tree representation of the formula

## Step 0: Get rid of → and ↔
## For ↔, do this intelligently, that is: if F↔G will be negated
## choose the suitable transformation to a DNF, (F∧G)∨(¬F∧¬G)
## otherwise go for (F→G)∧(G→F) (and, later, apply the standard
## transformation to the implications, of course)
## Note: this 'intelligence' is not required, of course, both
## transformations are semantically equivalent
def transform_step0(tree:Tree,neg=False,debug=0):
  # if debug: print("Node/Neg:",tree.c_typ,tree.c_val,neg)
  if tree.c_typ == "ATOM": return True
  if tree.c_typ == "1OP": return transform_step0(tree.left,not neg,debug)
  if tree.c_typ == "2OP":
    if tree.c_val == '↔':
      # if debug: print(tree,": Transform (F ↔ G)",end='')
      if debug: print("(Def.↔) Transform ",tree,end='')
      if not neg: # ->,<-
        # if debug: print(" to (F→G) ∧ (G→F)")
        new_node_l = Tree('→',parent=tree,left=tree.left,right=tree.right)
        new_node_r = Tree('→',parent=tree,left=copy.deepcopy(tree.right),
                                          right=copy.deepcopy(tree.left))
        tree.c_val = '∧'
      else:
        # if debug: print(" to (F∧G) ∨ (¬F∧¬G)")
        new_node_l = Tree('∧',parent=tree,left=tree.left,right=tree.right)
        new_node_r = Tree('∧',parent=tree,
                          left=Tree('¬',left=copy.deepcopy(tree.left)),
                          right=Tree('¬',left=copy.deepcopy(tree.right)))
        tree.c_val = '∨'
        
      # if debug: print("c_val changed to:",tree.c_val)
      tree.left = new_node_l
      tree.right = new_node_r
      if debug: print(tree)
      check_parents(tree,ancestor=tree.parent)
    elif tree.c_val == '→':
      # if debug: print(tree," Transform (F → G) to (NOT F OR G)")
      if debug: print("(Def.→) Transform ",tree,"to ",end='')
      # Create node with NEG left
      new_node = Tree('¬',parent=tree,left=tree.left)
      tree.c_val = '∨' # Change OP2 value to ∨
      tree.left = new_node
      if debug: print(tree)
      check_parents(tree,ancestor=tree.parent)
    return transform_step0(tree.left,neg,debug) \
              and transform_step0(tree.right,neg,debug)
  else:
    print("ERROR, unknown node type!")
    return False


""" print(res)
check_parents(res)
print(transform_step0(res,debug=0))
check_parents(res)
print(res)
 """
# Eval
def check2(t:Tree):
  print("Eval",t,":",t.eval({'A':False,'B':False}),",",end='')
  print(t.eval({'A':False,'B':True}),",",end='')
  print(t.eval({'A':True,'B':False}),",",end='')
  print(t.eval({'A':True,'B':True}))

def transform_step1(tree:Tree,debug=0):
  # if debug: print("Work on",tree,end='')
  # Get rid of double negation
  # if debug: print(" -- Node:",tree.c_typ,tree.c_val)
  if tree.c_typ == "ATOM": return True
  if tree.c_typ == "2OP":
    return transform_step1(tree.left,debug) and transform_step1(tree.right,debug)
  if tree.c_typ == "1OP":
    tl = tree.left
    if tl.c_typ == "1OP":
      # Doppelverneinung tree/NOT -> tl/NOT -> tl.left=child
      # Copy content from child to tree, except the parent,
      # which remains unchanged
      if debug: print("(Doppelverneinung) Transform ",tree,"to ",end='')
      child = tl.left
      tree.c_typ = child.c_typ
      tree.c_val = child.c_val
      tree.left = child.left
      tree.right = child.right
      if debug: print(tree)
      return transform_step1(tree,debug)
    elif tl.c_typ == "2OP":
      if debug: print("(DeMorgan) Transform ",tree,"to ",end='')
      # DeMorgan tree/Not -> tl/'∨,∧'
      if tl.c_val in ['∨','∧']:
        new_op = '∨' if tl.c_val == '∧' else '∧'
        tree.c_typ = "2OP"
        tree.c_val = new_op
        tree.left=Tree('¬',parent=tree,left=tl.left)
        tree.right=Tree('¬',parent=tree,left=tl.right)
        if debug: print(tree)
        return transform_step1(tree,debug)
      return transform_step1(tl,debug)
    else:
      return transform_step1(tl,debug)
  else:
    print("ERROR: UNKNOWN TYPE")
    return False

""" F = '((¬A ∨ B) → ¬(A ↔ B))'
res = create_tree(parse_formel(Tokenlist(tokenize(F)),""))
check2(res)
print("Formula:",res)
transform_step0(res,debug=0)
print("Schritt 0:",res)
check2(res)
transform_step1(res,debug=0)
print("Schritt 1:",res)
check2(res)
 """
## Note: this solution is for formulas with strictly correct syntax,
## this is ((AVB)VC) or (Av(BvC)) instead of (AvBvC).

def transform_step2(tree:Tree,debug=0):
  # if debug: print("Work on",tree,"(",tree.c_typ,"/",tree.c_val,")")
  if tree.c_typ == "ATOM": return True
  if tree.c_typ == "1OP": return transform_step2(tree.left,debug)
  if tree.c_typ == "2OP":
    op1 = tree.c_val
    if op1 == '∨':
      tl = tree.left
      tr = tree.right
      change = False
      if tl.c_typ == "2OP" and '∧' == tl.c_val:
        # if debug: print("  Transform ((G∧H)∨F) to ((F∨G)∧(F∨H))")
        # if debug: print("(Distributivgesetz) Transform (",tl,op1,tr,") to ((",tl.left,op1,tr,")",'∧',"(",tl.right,op1,tr,"))")
        if debug: print("(Distributivgesetz) Transform",tree," to ",end='')
        new_node_l = Tree(op1,parent=tree,left=tl.left,right=tree.right)
        new_node_r = Tree(op1,parent=tree,left=tl.right,
                                          right=copy.deepcopy(tree.right))
        change = True
      elif tr.c_typ == "2OP" and '∧' == tr.c_val:
        # if debug: print("  Transform (F∨(G∧H)) to ((F∨G)∧(F∨H))")
        # if debug: print("(Distributivgesetz)  Transform (",tl,op1,tr,") to ((",tl,op1,tr.left,")",'∧',"(",tl,op1,tr.right,"))")
        if debug: print("(Distributivgesetz) Transform",tree," to ",end='')
        new_node_l = Tree(op1,parent=tree,left=tree.left,right=tr.left)
        new_node_r = Tree(op1,parent=tree,left=copy.deepcopy(tree.left),right=tr.right)
        change = True
      if change:
        tree.c_val = '∧'
        tree.left = new_node_l
        tree.right = new_node_r
        if debug: print(tree)
        ## if you comment out the following if,
        ## an error will occur in some formulas, for example in
        ## F4 = (((¬C ∧ B) ∧ ((C ∧ A) ∨ (B ∧ (A → C)))) → (B ∨ (A → C)))
        ## because an OR is changed downstream of another OR to an AND,
        ## violating our structural condition: no AND below an OR
        ## To prevent this, we have to check the upstream operator after
        ## a change to an AND for being an OR (this may bubble up!)
        # if debug: print("Parent: ",tree.parent)
        if tree.parent and tree.parent.c_val == '∨':
          # if debug: print("Propagate change of OR upstream! \n  ",tree.parent)
          return transform_step2(tree.parent,debug)
    return transform_step2(tree.left,debug) \
              and transform_step2(tree.right,debug)
  else:
    print("ERROR, unknown node type!")
    return False
  
  
""" F = '((¬A ∨ B) → ¬(A ↔ B))'
res = create_tree(parse_formel(Tokenlist(tokenize(F)),""))
check2(res)
print("Formula:",res)
transform_step0(res,debug=0)
print("Schritt 0:",res)
check2(res)
transform_step1(res,debug=0)
print("Schritt 1:",res)
check2(res)
transform_step2(res,debug=0)
print("Schritt 2:",res)
check2(res) """

def perm(atoms):
    if len(atoms)==0: return [dict()]
    res1 = dict()
    res1[atoms[-1]] = False
    res2 = dict()
    res2[atoms[-1]] = True
    perms = perm(atoms[0:-1])
    res = []
    for p in perms:
      dic1 = {**p,**res1}
      dic2 = {**p,**res2}
      res.append(dic1)
      res.append(dic2)
    return res

# Eval the formula against its atoms
def check_(t:Tree,atoms,debug=0):
  perms = perm(atoms)
  # print(perms)
  print("Eval",t,": \n    ",end='')
  for p in perms:
    if debug:
      print(p,t.eval(p),",") # long output
    else:
      print(t.eval(p),",",end='')
  print("")


# We assume that the atom list is sorted
def check_markdown(t:Tree,atoms):
    out = ''

    # Table head
    for a in atoms:
        out += '|' + a 
    out += '|' + str(t) + "|  \n |"
    for i in range(len(atoms)):
        out += '-------- |'
    out += ':--------: |  \n'

    perms = perm(atoms)
    for p in perms:
        # Convert p to start of row in table^
        p_str = ''
        for e in p:
          val = 1 if p[e] else 0
          p_str += str(val) + "|"
        val = 1 if t.eval(p) else 0
        out += p_str + str(val)+" |  \n" # long output
    # print(out)
    return out


def process2(f,check=True,debug=0):
  res = create_tree(parse_formel(Tokenlist(tokenize(f)),""))
  check_parents(res)
  check2(res) if check else print("Formula: ",res)
  print("Step 0",transform_step0(res,debug=debug),":",res)
  check_parents(res)
  print("Step 1",transform_step1(res,debug=debug),":",res)
  check_parents(res)
  print("Step 2",transform_step2(res,debug=debug),":",res)
  check_parents(res)
  if check: check2(res)
  return res

def process(f,atoms,debug=0):
  res = create_tree(parse_formel(Tokenlist(tokenize(f)),""))
  check_(res,atoms)
  check_parents(res)
  print("Step 0",transform_step0(res,debug=debug),":",res)
  check_parents(res)
  print("Step 1",transform_step1(res,debug=debug),":",res)
  check_parents(res)
  print("Step 2",transform_step2(res,debug=debug),":",res)
  check_parents(res)
  check_(res,atoms)
  return res

""" F = '((¬A ∨ B) → ¬(A ↔ B))'
process2(F)

print("\nNext formula:")
F = '(¬((¬A → ¬B) ↔ ¬(A ↔ B))∨(B ↔ ¬(A ∧ ¬B)))'
process2(F,check=True)

print("\nNext formula:")
F = '((¬A ∧ ¬B) ∨ C)'
process(F,atoms=['A','B','C'])

print("\nNext formula:")
F = '(¬((¬A → ¬B) ↔ ¬(A → C)))'
process(F,atoms=['A','B','C']);
 """

def check_inner_negs(t:Tree):
  # First check fpr outer negations
  if t.c_typ == "ATOM": return True
  if t.c_typ == "1OP":
    return False if t.left.c_typ != "ATOM" else True
  if t.c_typ == "2OP":
    return check_inner_negs(t.left) and check_inner_negs(t.right)

def check_no_arrows(t:Tree):
  if t.c_typ == "ATOM": return True
  if t.c_typ == "1OP":
    return check_no_arrows(t.left)
  if t.c_typ == "2OP":
    if t.c_val not in ['∨','∧']: return False
    return check_no_arrows(t.left) and check_no_arrows(t.right)

def check_and_of_ors(t:Tree,debug=0,pre=0):
  if debug: print(pre*'  ',"Check OP:",t.c_typ,"/",t.c_val)
  if t.c_typ == "ATOM": return True
  if t.c_typ == "1OP":
    return check_and_of_ors(t.left,debug=debug,pre=pre+1)
  if t.c_typ == "2OP":
    if debug: print(pre*'  ',t.c_val,":",t,"\n  Left:",t.left,"|",t.left.c_val,"\n Right:",t.right,"|",t.right.c_val)
    if t.c_val == '∨':
      if t.left.c_val == '∧' or t.right.c_val == '∧':
        if debug: print(pre*'  ',"Error in",t.left,"|",t.left.c_val,t.c_val,t.right,"|",t.right.c_val)
        return False
    return check_and_of_ors(t.left,debug=debug,pre=pre+1) and check_and_of_ors(t.right,debug=debug,pre=pre+1)

""" f = '(¬(C ∨ ((A ↔ (A ↔ C)) → B)) ∧ (((A ∧ C) ∧ B) ∨ (B → A)))'
res = create_tree(parse_formel(Tokenlist(tokenize(f)),""))
print(f)
print("Do we have only innermost negations?",check_inner_negs(res))
print("Do we have only have ANDs and ORs?",check_no_arrows(res))
print("Are ORs only parents of other ORs?",check_and_of_ors(res),"\n")

res = process(f,atoms=['A','B','C'],debug=0)
print("Do we have only innermost negations?",check_inner_negs(res))
print("Do we have only have ANDs and ORs?",check_no_arrows(res))
print("Are ORs only parents of other ORs?",check_and_of_ors(res))
 """

""" F1 = '((¬B ↔ A) ↔ (((C ∨ B) → ((B → C) ↔ ¬A)) ↔ (B ∨ A)))'
F2 = '((B → C) → ((((C ∧ (A ∧ B)) ↔ (A ∨ C)) ∨ (B ∨ B)) ↔ (C ↔ A)))'
F3 = '((C ∧ A) ∨ ((C → B) ∨ (¬(A → A) ↔ ((¬C ∧ B) ∨ B))))'
F4 = '(((¬C ∧ B) ∧ ((C ∧ A) ∨ (B ∧ (A → C)))) → (B ∨ (A → C)))'
F5 = '¬(((B ∨ (C ↔ A)) ↔ (A ∨ (B ↔ C))) ∨ (A ∨ (C → B)))'
F6 = '¬((¬(¬C ∧ (B → C)) ∧ ¬A) ↔ ((C ∨ B) → A))'
F7 = '(¬(C ∨ ((A ↔ (A ↔ C)) → B)) ∧ (((A ∧ C) ∧ B) ∨ (B → A)))'

F8 = '(((¬C∧(¬B∨C))∨A)∨((C∨B)∧¬A))'

## Oder in ausführlicherer Form:
formulas = [F1,F2,F3,F4,F5,F6,F7,F8]
# formulas = [F6,F8]
for f in formulas:
  print(f)
  res = process(f,atoms=['A','B','C'],debug=0)
  # res = process(f,atoms=['A','B','C'],debug=1)
  print("Do we have only innermost negations?",check_inner_negs(res))
  print("Check parent?",check_parents(res))
  print("Do we have only have ANDs and ORs?",check_no_arrows(res))
  print("Check parent?",check_parents(res))
  print("Are ORs only parents of other ORs?",check_and_of_ors(res,debug=0))
  print("Check parent?",check_parents(res))
  print("") """

def reduce_braces(t:Tree,first=False):
  '''
  ((((¬C∨A)∨(C∨B))∧(((¬B∨C)∨A)∨(C∨B)))∧(((¬C∨A)∨¬A)∧(((¬B∨C)∨A)∨¬A)))
  What is the idea of this reduction?
  We already know that we have no ANDs below an OR.
  We only set braces when we start with a new "top-level" OR.
  (¬C∨A∨C∨B)∧(¬B∨C∨A∨C∨B)∧(¬C∨A∨¬A)∧(¬B∨C∨A∨¬A)
  '''
  if t.c_typ == "ATOM": return t.c_val
  if t.c_typ == "1OP":
    return t.c_val + reduce_braces(t.left)
  if t.c_typ == "2OP":
    if t.c_val == '∨':
      res = reduce_braces(t.left) + '∨' + reduce_braces(t.right)
      return "(" + res + ")" if first else res
    if t.c_val == '∧':
      return reduce_braces(t.left,True) + '∧' + reduce_braces(t.right,True)
    print("ERROR: Formula",t,"is not in KNF!")

# print(reduce_braces(res,True))

""" 
## Complete flow for a formula
F = "¬((¬(¬C ∧ (B → C)) ∧ ¬A) → ((C ∨ B) → A))"
res = process(F,atoms=['A','B','C'],debug=0)
print("\nKNF:", reduce_braces(res,True),"\n\n")

F = "(((¬C∧(¬B∨C))∨A)→((C∨B)∧¬A))"
res = process(F,atoms=['A','B','C'],debug=0)
print("\nKNF:", reduce_braces(res,True)) """