import streamlit as st
import knf
import io
from contextlib import redirect_stdout
import time
import re

version = 0.815

st.markdown("##### Produce CNFs from propositional logic formulas! Version " + str(version) + " (Jan.2024)")

st.write("You may use ¬,→,↔,∧,∨  or -,->,<->,n,v and A,B,C,D,E and, of course, (,)")

instr = '¬((¬(¬C ∧ (B → C)) ∧ ¬A) → ((C ∨ B) → A))'
# instr = "Please, enter a formula!"

with st.form('formula_input_form'):
    # Create two columns; adjust the ratio to your liking
    col1, col2 = st.columns([9,2]) 

    # Use the first column for text input
    with col1:
        F = st.text_input(
            instr,
            key='formula',
            value=instr,
            placeholder="Enter a formula...",
            label_visibility='collapsed'
        )
    
        debug = st.checkbox("With details?")
        
    # Use the second column for the submit button
    with col2:
        submitted = st.form_submit_button('CNFify')

    if F and submitted:
        # Do something with the inputted text here
        ret,F,atoms = knf.preprocess(F)
        st.write("Normalized Input: ",F)
        if not ret: st.write("ERROR: "+F)
        else:
            # debug = True
            with io.StringIO() as buf, redirect_stdout(buf):
                # state,res = knf.process(F,atoms=atoms,debug=False)
                res = knf.create_tree(knf.parse_formel(knf.Tokenlist(knf.tokenize(F)),""))
                knf.check_(res,atoms)
                knf.check_parents(res)
                print("\n")
                if debug: print("**Step 0: get rid of → and ↔:**")
                print("Step 0",knf.transform_step0(res,debug=debug),":",res)
                knf.check_parents(res)
                if debug: print("\n")
                if debug: print("**Step 1: move ¬ inwards, remove ¬¬:**")
                print("Step 1",knf.transform_step1(res,debug=debug),":",res)
                knf.check_parents(res)
                if debug: print("\n")
                if debug: print("**Step 2: move ∨ inwards using distributivity:**")
                print("Step 2",knf.transform_step2(res,debug=debug),":",res)
                knf.check_parents(res)
                print("\n")
                knf.check_(res,atoms)
                output = buf.getvalue()

                # st.write(output)
                st.write("\nKNF: "+knf.reduce_braces(res,True)+"\n")

                if debug:
                    eval_str = knf.check_markdown(res,atoms)
                    output += eval_str + "  \n"

                with st.chat_message("assistant"):
                    m = st.empty()
                    # Simulate stream of response with milliseconds delay
                    full_response = ""
                    # for chunk in re.split(r'(\s+)', output):
                    for chunk in output.splitlines():
                        full_response += chunk + "  \n"
                        time.sleep(0.01)

                        # Add a blinking cursor to simulate typing
                        m.markdown(full_response + "▌")

if st.button("Help"):
    st.write("Hints: Be correct with Klammern...we expect the formulas to " \
         + "obey the syntax definition:\n" \
         + "Every binary operator requires parenthesis, that is (A v B v C)\n" \
         + "should be ((AvB)vC) or (Av(BvC)).  \n" \
         + "Remark: If you leave out some (,), it will still work somehow most of the time\n" + \
         "but you need to make sure yourself that the result is meaningful.")
    st.write("To keep it simple, we add outer parenthesis in any case (as they are often forgotten)")
