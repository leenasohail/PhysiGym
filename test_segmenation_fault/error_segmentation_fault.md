# Problem: error segmentation fault
## How to generate the error:
How to generate a dump file:
```
ulimit -c unlimited
```
Set Unlimited Core Dump Size
```
 echo "your_chosen_path/newest_core.%e.%p" | sudo tee /proc/sys/kernel/core_pattern
```
your_chosen_path where the core dump is saved.

And then launch test_env.py.

## The error :
Use gdb, and you can have this message:
```
0x0000729d6a42e834 in BioFVM::operator+= (v1=..., 
    v2=std::vector of length 1, capacity 1 = {...})
    at BioFVM/BioFVM_vector.cpp:133
        i = 0
#1  0x0000729d6a411a62 in BioFVM::Basic_Agent::release_internalized_substrates (this=this@entry=0x635f0eaeba40)
    at BioFVM/BioFVM_basic_agent.cpp:222
        pS = 0x729d6a4f9100 <BioFVM::microenvironment>
#2  0x0000729d6a4547fb in PhysiCell::delete_cell (index=35)
    at core/PhysiCell_cell.cpp:1155
        pDeleteMe = 0x635f0eaeba40
#3  0x0000729d6a45489f in PhysiCell::delete_cell (
    pDelete=<optimized out>) at core/PhysiCell_cell.cpp:1206
No locals.
#4  0x0000729d6a4548a9 in PhysiCell::Cell::die (
    this=<optimized out>) at core/PhysiCell_cell.cpp:832
No locals.
#5  0x0000729d6a409f2c in generate_cell_types () at ../custom.cpp:41
        pCell = <optimized out>
        __for_range = <optimized out>
        __for_begin = <optimized out>
        __for_end = <optimized out>
#6  0x0000729d6a4c7c75 in physicell_start (self=<optimized out>, 
    args=<optimized out>) at physicellmodule.cpp:91
        argument = 0x729d67206130 "config/PhysiCell_settings.xml"
        XML_status = <optimized out>
        copy_command = "cp config/PhysiCell_settings.xml output\000\000\000\000\000\000\000\000\000\000\060\000\000\000\000\000\000\000\02--Type <RET> for more, q to quit, c to continue without paging--
0\000\000\000\000\000\000\240@\257\a_c\000\000\360\256D\250\235r\000\000\240lD\250\235r\000\000\310N\312\312\375\177\000\000X\"\\\250\235r\000\000{\300k\a_c\000\000\360\344w0\000\000\000\000p\033\217\b_c\000\000\020um\a_c\000\000p\222\\\250\235r\000\000@\003Mf\235r\000\000h\"\\\250\235r\000\000\000\263\255\a_c\000\000=Nm\a_c\000\000\000\000\000\000\000\000\000\000\352\236}\a_c\000\000\a\000\000\000\000\000\000\000"...
        time_units = "min"
        mechanics_voxel_size = 30
        cell_container = <optimized out>
        cell_coloring_function = <optimized out>
#7  0x0000635f076d8138 in ?? ()
No symbol table info available.
#8  0x0000635f076cea7b in _PyObject_MakeTpCall ()
No symbol table info available.
#9  0x0000635f076c7629 in _PyEval_EvalFrameDefault ()
No symbol table info available.
#10 0x0000635f076e6a51 in ?? ()
No symbol table info available.
#11 0x0000635f076c35d7 in _PyEval_EvalFrameDefault ()
No symbol table info available.
#12 0x0000635f076e6a51 in ?? ()
No symbol table info available.
#13 0x0000635f076c35d7 in _PyEval_EvalFrameDefault ()
No symbol table info available.
#14 0x0000635f076e67f1 in ?? ()
No symbol table info available.
#15 0x0000635f076c6cfa in _PyEval_EvalFrameDefault ()
--Type <RET> for more, q to quit, c to continue without paging--
No symbol table info available.
#16 0x0000635f076bd9c6 in ?? ()
No symbol table info available.
#17 0x0000635f077b3256 in PyEval_EvalCode ()
No symbol table info available.
#18 0x0000635f077de108 in ?? ()
No symbol table info available.
#19 0x0000635f077d79cb in ?? ()
No symbol table info available.
#20 0x0000635f077dde55 in ?? ()
No symbol table info available.
#21 0x0000635f077dd338 in _PyRun_SimpleFileObject ()
No symbol table info available.
#22 0x0000635f077dcf83 in _PyRun_AnyFileObject ()
No symbol table info available.
#23 0x0000635f077cfa5e in Py_RunMain ()
No symbol table info available.
#24 0x0000635f077a602d in Py_BytesMain ()
No symbol table info available.
#25 0x0000729da8e29d90 in __libc_start_call_main (
    main=main@entry=0x635f077a5ff0, argc=argc@entry=2, 
    argv=argv@entry=0x7ffdcaca5e78) at ../sysdeps/nptl/libc_start_call_main.h:58
        self = <optimized out>
        result = <optimized out>
        unwind_buf = {cancel_jmp_buf = {{jmp_buf = {0, -6233043386193527981, 
                140728005713528, 109259798503408, 109259801864248, 
                126021473202240, 6234210608971135827, 5495686586527529811}, 
              mask_was_saved = 0}}, priv = {pad = {0x0, 0x0, 0x0, 0x0}, data = {
              prev = 0x0, cleanup = 0x0, canceltype = 0}}}
        not_first_call = <optimized out>
#26 0x0000729da8e29e40 in __libc_start_main_impl (main=0x635f077a5ff0, argc=2, 
    argv=0x7ffdcaca5e78, init=<optimized out>, fini=<optimized out>, 
    rtld_fini=<optimized out>, stack_end=0x7ffdcaca5e68)
    at ../csu/libc-start.c:392
```
when i do v1, it is said v1 is optimized out, we do not acces to the v1.size(), while for v2 , we have the size of 1.

If you go deeper, you can get access to the assembly code, and find where the error occurs.
```
(gdb) disassemble
Dump of assembler code for function _ZN6BioFVMpLERSt6vectorIdSaIdEERKS2_:
   0x0000794281a2e810 <+0>:     endbr64 
   0x0000794281a2e814 <+4>:     mov    (%rdi),%r8
   0x0000794281a2e817 <+7>:     mov    0x8(%rdi),%rdi
   0x0000794281a2e81b <+11>:    sub    %r8,%rdi
   0x0000794281a2e81e <+14>:    sar    $0x3,%rdi
   0x0000794281a2e822 <+18>:    je     0x794281a2e84c <_ZN6BioFVMpLERSt6vectorIdSaIdEERKS2_+60>
   0x0000794281a2e824 <+20>:    mov    (%rsi),%rsi
   0x0000794281a2e827 <+23>:    xor    %edx,%edx
   0x0000794281a2e829 <+25>:    xor    %eax,%eax
   0x0000794281a2e82b <+27>:    nopl   0x0(%rax,%rax,1)
   0x0000794281a2e830 <+32>:    lea    (%r8,%rax,8),%rcx
=> 0x0000794281a2e834 <+36>:    vmovsd (%rcx),%xmm0
   0x0000794281a2e838 <+40>:    vaddsd (%rsi,%rax,8),%xmm0,%xmm0
   0x0000794281a2e83d <+45>:    lea    0x1(%rdx),%eax
--Type <RET> for more, q to quit, c to continue without paging--
   0x0000794281a2e840 <+48>:    mov    %rax,%rdx
   0x0000794281a2e843 <+51>:    vmovsd %xmm0,(%rcx)
   0x0000794281a2e847 <+55>:    cmp    %rdi,%rax
   0x0000794281a2e84a <+58>:    jb     0x794281a2e830 <_ZN6BioFVMpLERSt6vectorIdSaIdEERKS2_+32>
   0x0000794281a2e84c <+60>:    ret  
```

## According to ChatGPT:
The issue seems to be related to the operation on the v1 vector in the BioFVM::operator+= function, specifically at this line:

cpp
Copier le code
{ v1[i] += v2[i]; }
Understanding the Problem
Segmentation Fault: The crash happens because of a segmentation fault, which usually occurs when the program tries to access memory that it shouldn't, such as dereferencing a null or invalid pointer, or accessing an out-of-bounds element in a vector or array.

Registers and Assembly Analysis:

The instruction causing the crash is vmovsd (%rcx),%xmm0, which moves data from memory (address in rcx) to an XMM register.
The rcx register holds the address calculated by lea (%r8,%rax,8),%rcx, which adds an offset to the base address r8 to get the address of the i-th element of v1.
If r8 is invalid or points to an invalid memory region, dereferencing it to access v1[i] will cause a segmentation fault.
v1 is "optimized out": In your debugging session, you tried to inspect v1 but found it was "optimized out." This means that the compiler optimized away the variable's explicit storage in a way that makes it inaccessible to the debugger. However, the issue might still lie with how v1 is accessed or initialized.

Potential Causes of the Problem
Invalid Memory Access:

v1 might not be correctly initialized, or it might be pointing to an invalid memory region. This could happen if v1 was created but then reallocated, leaving the pointer stale.
Out-of-Bounds Access:

The loop or operation involving v1 might be trying to access an element beyond the size of the vector, leading to undefined behavior.
Memory Corruption:

There could be memory corruption elsewhere in the program that affected the memory area where v1 is stored.
