
Í
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
º
	MLCMatMul
a"T
b"T

unique_key"T*num_args
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2"
num_argsint ("

input_rankint(0
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
¾
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*	2.4.0-rc02v1.12.1-44683-gbcaa5ccc43e8ä
|
dense_627/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_627/kernel
u
$dense_627/kernel/Read/ReadVariableOpReadVariableOpdense_627/kernel*
_output_shapes

:*
dtype0
t
dense_627/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_627/bias
m
"dense_627/bias/Read/ReadVariableOpReadVariableOpdense_627/bias*
_output_shapes
:*
dtype0
|
dense_628/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_628/kernel
u
$dense_628/kernel/Read/ReadVariableOpReadVariableOpdense_628/kernel*
_output_shapes

:*
dtype0
t
dense_628/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_628/bias
m
"dense_628/bias/Read/ReadVariableOpReadVariableOpdense_628/bias*
_output_shapes
:*
dtype0
|
dense_629/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_629/kernel
u
$dense_629/kernel/Read/ReadVariableOpReadVariableOpdense_629/kernel*
_output_shapes

:*
dtype0
t
dense_629/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_629/bias
m
"dense_629/bias/Read/ReadVariableOpReadVariableOpdense_629/bias*
_output_shapes
:*
dtype0
|
dense_630/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_630/kernel
u
$dense_630/kernel/Read/ReadVariableOpReadVariableOpdense_630/kernel*
_output_shapes

:*
dtype0
t
dense_630/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_630/bias
m
"dense_630/bias/Read/ReadVariableOpReadVariableOpdense_630/bias*
_output_shapes
:*
dtype0
|
dense_631/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_631/kernel
u
$dense_631/kernel/Read/ReadVariableOpReadVariableOpdense_631/kernel*
_output_shapes

:*
dtype0
t
dense_631/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_631/bias
m
"dense_631/bias/Read/ReadVariableOpReadVariableOpdense_631/bias*
_output_shapes
:*
dtype0
|
dense_632/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_632/kernel
u
$dense_632/kernel/Read/ReadVariableOpReadVariableOpdense_632/kernel*
_output_shapes

:*
dtype0
t
dense_632/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_632/bias
m
"dense_632/bias/Read/ReadVariableOpReadVariableOpdense_632/bias*
_output_shapes
:*
dtype0
|
dense_633/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_633/kernel
u
$dense_633/kernel/Read/ReadVariableOpReadVariableOpdense_633/kernel*
_output_shapes

:*
dtype0
t
dense_633/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_633/bias
m
"dense_633/bias/Read/ReadVariableOpReadVariableOpdense_633/bias*
_output_shapes
:*
dtype0
|
dense_634/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_634/kernel
u
$dense_634/kernel/Read/ReadVariableOpReadVariableOpdense_634/kernel*
_output_shapes

:*
dtype0
t
dense_634/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_634/bias
m
"dense_634/bias/Read/ReadVariableOpReadVariableOpdense_634/bias*
_output_shapes
:*
dtype0
|
dense_635/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_635/kernel
u
$dense_635/kernel/Read/ReadVariableOpReadVariableOpdense_635/kernel*
_output_shapes

:*
dtype0
t
dense_635/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_635/bias
m
"dense_635/bias/Read/ReadVariableOpReadVariableOpdense_635/bias*
_output_shapes
:*
dtype0
|
dense_636/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_636/kernel
u
$dense_636/kernel/Read/ReadVariableOpReadVariableOpdense_636/kernel*
_output_shapes

:*
dtype0
t
dense_636/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_636/bias
m
"dense_636/bias/Read/ReadVariableOpReadVariableOpdense_636/bias*
_output_shapes
:*
dtype0
|
dense_637/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_637/kernel
u
$dense_637/kernel/Read/ReadVariableOpReadVariableOpdense_637/kernel*
_output_shapes

:*
dtype0
t
dense_637/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_637/bias
m
"dense_637/bias/Read/ReadVariableOpReadVariableOpdense_637/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0

Adam/dense_627/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_627/kernel/m

+Adam/dense_627/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_627/kernel/m*
_output_shapes

:*
dtype0

Adam/dense_627/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_627/bias/m
{
)Adam/dense_627/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_627/bias/m*
_output_shapes
:*
dtype0

Adam/dense_628/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_628/kernel/m

+Adam/dense_628/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_628/kernel/m*
_output_shapes

:*
dtype0

Adam/dense_628/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_628/bias/m
{
)Adam/dense_628/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_628/bias/m*
_output_shapes
:*
dtype0

Adam/dense_629/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_629/kernel/m

+Adam/dense_629/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_629/kernel/m*
_output_shapes

:*
dtype0

Adam/dense_629/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_629/bias/m
{
)Adam/dense_629/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_629/bias/m*
_output_shapes
:*
dtype0

Adam/dense_630/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_630/kernel/m

+Adam/dense_630/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_630/kernel/m*
_output_shapes

:*
dtype0

Adam/dense_630/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_630/bias/m
{
)Adam/dense_630/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_630/bias/m*
_output_shapes
:*
dtype0

Adam/dense_631/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_631/kernel/m

+Adam/dense_631/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_631/kernel/m*
_output_shapes

:*
dtype0

Adam/dense_631/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_631/bias/m
{
)Adam/dense_631/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_631/bias/m*
_output_shapes
:*
dtype0

Adam/dense_632/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_632/kernel/m

+Adam/dense_632/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_632/kernel/m*
_output_shapes

:*
dtype0

Adam/dense_632/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_632/bias/m
{
)Adam/dense_632/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_632/bias/m*
_output_shapes
:*
dtype0

Adam/dense_633/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_633/kernel/m

+Adam/dense_633/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_633/kernel/m*
_output_shapes

:*
dtype0

Adam/dense_633/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_633/bias/m
{
)Adam/dense_633/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_633/bias/m*
_output_shapes
:*
dtype0

Adam/dense_634/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_634/kernel/m

+Adam/dense_634/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_634/kernel/m*
_output_shapes

:*
dtype0

Adam/dense_634/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_634/bias/m
{
)Adam/dense_634/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_634/bias/m*
_output_shapes
:*
dtype0

Adam/dense_635/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_635/kernel/m

+Adam/dense_635/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_635/kernel/m*
_output_shapes

:*
dtype0

Adam/dense_635/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_635/bias/m
{
)Adam/dense_635/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_635/bias/m*
_output_shapes
:*
dtype0

Adam/dense_636/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_636/kernel/m

+Adam/dense_636/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_636/kernel/m*
_output_shapes

:*
dtype0

Adam/dense_636/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_636/bias/m
{
)Adam/dense_636/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_636/bias/m*
_output_shapes
:*
dtype0

Adam/dense_637/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_637/kernel/m

+Adam/dense_637/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_637/kernel/m*
_output_shapes

:*
dtype0

Adam/dense_637/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_637/bias/m
{
)Adam/dense_637/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_637/bias/m*
_output_shapes
:*
dtype0

Adam/dense_627/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_627/kernel/v

+Adam/dense_627/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_627/kernel/v*
_output_shapes

:*
dtype0

Adam/dense_627/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_627/bias/v
{
)Adam/dense_627/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_627/bias/v*
_output_shapes
:*
dtype0

Adam/dense_628/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_628/kernel/v

+Adam/dense_628/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_628/kernel/v*
_output_shapes

:*
dtype0

Adam/dense_628/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_628/bias/v
{
)Adam/dense_628/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_628/bias/v*
_output_shapes
:*
dtype0

Adam/dense_629/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_629/kernel/v

+Adam/dense_629/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_629/kernel/v*
_output_shapes

:*
dtype0

Adam/dense_629/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_629/bias/v
{
)Adam/dense_629/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_629/bias/v*
_output_shapes
:*
dtype0

Adam/dense_630/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_630/kernel/v

+Adam/dense_630/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_630/kernel/v*
_output_shapes

:*
dtype0

Adam/dense_630/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_630/bias/v
{
)Adam/dense_630/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_630/bias/v*
_output_shapes
:*
dtype0

Adam/dense_631/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_631/kernel/v

+Adam/dense_631/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_631/kernel/v*
_output_shapes

:*
dtype0

Adam/dense_631/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_631/bias/v
{
)Adam/dense_631/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_631/bias/v*
_output_shapes
:*
dtype0

Adam/dense_632/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_632/kernel/v

+Adam/dense_632/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_632/kernel/v*
_output_shapes

:*
dtype0

Adam/dense_632/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_632/bias/v
{
)Adam/dense_632/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_632/bias/v*
_output_shapes
:*
dtype0

Adam/dense_633/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_633/kernel/v

+Adam/dense_633/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_633/kernel/v*
_output_shapes

:*
dtype0

Adam/dense_633/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_633/bias/v
{
)Adam/dense_633/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_633/bias/v*
_output_shapes
:*
dtype0

Adam/dense_634/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_634/kernel/v

+Adam/dense_634/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_634/kernel/v*
_output_shapes

:*
dtype0

Adam/dense_634/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_634/bias/v
{
)Adam/dense_634/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_634/bias/v*
_output_shapes
:*
dtype0

Adam/dense_635/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_635/kernel/v

+Adam/dense_635/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_635/kernel/v*
_output_shapes

:*
dtype0

Adam/dense_635/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_635/bias/v
{
)Adam/dense_635/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_635/bias/v*
_output_shapes
:*
dtype0

Adam/dense_636/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_636/kernel/v

+Adam/dense_636/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_636/kernel/v*
_output_shapes

:*
dtype0

Adam/dense_636/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_636/bias/v
{
)Adam/dense_636/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_636/bias/v*
_output_shapes
:*
dtype0

Adam/dense_637/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_637/kernel/v

+Adam/dense_637/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_637/kernel/v*
_output_shapes

:*
dtype0

Adam/dense_637/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_637/bias/v
{
)Adam/dense_637/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_637/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
Æj
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*j
value÷iBôi Bíi
 
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
layer_with_weights-5
layer-5
layer_with_weights-6
layer-6
layer_with_weights-7
layer-7
	layer_with_weights-8
	layer-8

layer_with_weights-9

layer-9
layer_with_weights-10
layer-10
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api

signatures
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
 trainable_variables
!	variables
"regularization_losses
#	keras_api
h

$kernel
%bias
&trainable_variables
'	variables
(regularization_losses
)	keras_api
h

*kernel
+bias
,trainable_variables
-	variables
.regularization_losses
/	keras_api
h

0kernel
1bias
2trainable_variables
3	variables
4regularization_losses
5	keras_api
h

6kernel
7bias
8trainable_variables
9	variables
:regularization_losses
;	keras_api
h

<kernel
=bias
>trainable_variables
?	variables
@regularization_losses
A	keras_api
h

Bkernel
Cbias
Dtrainable_variables
E	variables
Fregularization_losses
G	keras_api
h

Hkernel
Ibias
Jtrainable_variables
K	variables
Lregularization_losses
M	keras_api
h

Nkernel
Obias
Ptrainable_variables
Q	variables
Rregularization_losses
S	keras_api
ø
Titer

Ubeta_1

Vbeta_2
	Wdecay
Xlearning_ratemmmmmm$m %m¡*m¢+m£0m¤1m¥6m¦7m§<m¨=m©BmªCm«Hm¬Im­Nm®Om¯v°v±v²v³v´vµ$v¶%v·*v¸+v¹0vº1v»6v¼7v½<v¾=v¿BvÀCvÁHvÂIvÃNvÄOvÅ
¦
0
1
2
3
4
5
$6
%7
*8
+9
010
111
612
713
<14
=15
B16
C17
H18
I19
N20
O21
¦
0
1
2
3
4
5
$6
%7
*8
+9
010
111
612
713
<14
=15
B16
C17
H18
I19
N20
O21
 
­
Ynon_trainable_variables
trainable_variables

Zlayers
	variables
[metrics
regularization_losses
\layer_metrics
]layer_regularization_losses
 
\Z
VARIABLE_VALUEdense_627/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_627/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
­
^non_trainable_variables
trainable_variables

_layers
	variables
`metrics
regularization_losses
alayer_metrics
blayer_regularization_losses
\Z
VARIABLE_VALUEdense_628/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_628/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
­
cnon_trainable_variables
trainable_variables

dlayers
	variables
emetrics
regularization_losses
flayer_metrics
glayer_regularization_losses
\Z
VARIABLE_VALUEdense_629/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_629/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
­
hnon_trainable_variables
 trainable_variables

ilayers
!	variables
jmetrics
"regularization_losses
klayer_metrics
llayer_regularization_losses
\Z
VARIABLE_VALUEdense_630/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_630/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

$0
%1

$0
%1
 
­
mnon_trainable_variables
&trainable_variables

nlayers
'	variables
ometrics
(regularization_losses
player_metrics
qlayer_regularization_losses
\Z
VARIABLE_VALUEdense_631/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_631/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

*0
+1

*0
+1
 
­
rnon_trainable_variables
,trainable_variables

slayers
-	variables
tmetrics
.regularization_losses
ulayer_metrics
vlayer_regularization_losses
\Z
VARIABLE_VALUEdense_632/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_632/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

00
11

00
11
 
­
wnon_trainable_variables
2trainable_variables

xlayers
3	variables
ymetrics
4regularization_losses
zlayer_metrics
{layer_regularization_losses
\Z
VARIABLE_VALUEdense_633/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_633/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

60
71

60
71
 
®
|non_trainable_variables
8trainable_variables

}layers
9	variables
~metrics
:regularization_losses
layer_metrics
 layer_regularization_losses
\Z
VARIABLE_VALUEdense_634/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_634/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

<0
=1

<0
=1
 
²
non_trainable_variables
>trainable_variables
layers
?	variables
metrics
@regularization_losses
layer_metrics
 layer_regularization_losses
\Z
VARIABLE_VALUEdense_635/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_635/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

B0
C1

B0
C1
 
²
non_trainable_variables
Dtrainable_variables
layers
E	variables
metrics
Fregularization_losses
layer_metrics
 layer_regularization_losses
\Z
VARIABLE_VALUEdense_636/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_636/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE

H0
I1

H0
I1
 
²
non_trainable_variables
Jtrainable_variables
layers
K	variables
metrics
Lregularization_losses
layer_metrics
 layer_regularization_losses
][
VARIABLE_VALUEdense_637/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEdense_637/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE

N0
O1

N0
O1
 
²
non_trainable_variables
Ptrainable_variables
layers
Q	variables
metrics
Rregularization_losses
layer_metrics
 layer_regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
N
0
1
2
3
4
5
6
7
	8

9
10

0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

total

count
	variables
	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

0
1

	variables
}
VARIABLE_VALUEAdam/dense_627/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_627/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_628/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_628/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_629/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_629/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_630/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_630/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_631/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_631/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_632/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_632/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_633/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_633/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_634/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_634/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_635/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_635/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_636/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_636/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/dense_637/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense_637/bias/mQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_627/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_627/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_628/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_628/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_629/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_629/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_630/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_630/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_631/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_631/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_632/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_632/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_633/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_633/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_634/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_634/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_635/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_635/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_636/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_636/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/dense_637/kernel/vSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/dense_637/bias/vQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_dense_627_inputPlaceholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
Þ
StatefulPartitionedCallStatefulPartitionedCallserving_default_dense_627_inputdense_627/kerneldense_627/biasdense_628/kerneldense_628/biasdense_629/kerneldense_629/biasdense_630/kerneldense_630/biasdense_631/kerneldense_631/biasdense_632/kerneldense_632/biasdense_633/kerneldense_633/biasdense_634/kerneldense_634/biasdense_635/kerneldense_635/biasdense_636/kerneldense_636/biasdense_637/kerneldense_637/bias*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 */
f*R(
&__inference_signature_wrapper_15195866
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$dense_627/kernel/Read/ReadVariableOp"dense_627/bias/Read/ReadVariableOp$dense_628/kernel/Read/ReadVariableOp"dense_628/bias/Read/ReadVariableOp$dense_629/kernel/Read/ReadVariableOp"dense_629/bias/Read/ReadVariableOp$dense_630/kernel/Read/ReadVariableOp"dense_630/bias/Read/ReadVariableOp$dense_631/kernel/Read/ReadVariableOp"dense_631/bias/Read/ReadVariableOp$dense_632/kernel/Read/ReadVariableOp"dense_632/bias/Read/ReadVariableOp$dense_633/kernel/Read/ReadVariableOp"dense_633/bias/Read/ReadVariableOp$dense_634/kernel/Read/ReadVariableOp"dense_634/bias/Read/ReadVariableOp$dense_635/kernel/Read/ReadVariableOp"dense_635/bias/Read/ReadVariableOp$dense_636/kernel/Read/ReadVariableOp"dense_636/bias/Read/ReadVariableOp$dense_637/kernel/Read/ReadVariableOp"dense_637/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp+Adam/dense_627/kernel/m/Read/ReadVariableOp)Adam/dense_627/bias/m/Read/ReadVariableOp+Adam/dense_628/kernel/m/Read/ReadVariableOp)Adam/dense_628/bias/m/Read/ReadVariableOp+Adam/dense_629/kernel/m/Read/ReadVariableOp)Adam/dense_629/bias/m/Read/ReadVariableOp+Adam/dense_630/kernel/m/Read/ReadVariableOp)Adam/dense_630/bias/m/Read/ReadVariableOp+Adam/dense_631/kernel/m/Read/ReadVariableOp)Adam/dense_631/bias/m/Read/ReadVariableOp+Adam/dense_632/kernel/m/Read/ReadVariableOp)Adam/dense_632/bias/m/Read/ReadVariableOp+Adam/dense_633/kernel/m/Read/ReadVariableOp)Adam/dense_633/bias/m/Read/ReadVariableOp+Adam/dense_634/kernel/m/Read/ReadVariableOp)Adam/dense_634/bias/m/Read/ReadVariableOp+Adam/dense_635/kernel/m/Read/ReadVariableOp)Adam/dense_635/bias/m/Read/ReadVariableOp+Adam/dense_636/kernel/m/Read/ReadVariableOp)Adam/dense_636/bias/m/Read/ReadVariableOp+Adam/dense_637/kernel/m/Read/ReadVariableOp)Adam/dense_637/bias/m/Read/ReadVariableOp+Adam/dense_627/kernel/v/Read/ReadVariableOp)Adam/dense_627/bias/v/Read/ReadVariableOp+Adam/dense_628/kernel/v/Read/ReadVariableOp)Adam/dense_628/bias/v/Read/ReadVariableOp+Adam/dense_629/kernel/v/Read/ReadVariableOp)Adam/dense_629/bias/v/Read/ReadVariableOp+Adam/dense_630/kernel/v/Read/ReadVariableOp)Adam/dense_630/bias/v/Read/ReadVariableOp+Adam/dense_631/kernel/v/Read/ReadVariableOp)Adam/dense_631/bias/v/Read/ReadVariableOp+Adam/dense_632/kernel/v/Read/ReadVariableOp)Adam/dense_632/bias/v/Read/ReadVariableOp+Adam/dense_633/kernel/v/Read/ReadVariableOp)Adam/dense_633/bias/v/Read/ReadVariableOp+Adam/dense_634/kernel/v/Read/ReadVariableOp)Adam/dense_634/bias/v/Read/ReadVariableOp+Adam/dense_635/kernel/v/Read/ReadVariableOp)Adam/dense_635/bias/v/Read/ReadVariableOp+Adam/dense_636/kernel/v/Read/ReadVariableOp)Adam/dense_636/bias/v/Read/ReadVariableOp+Adam/dense_637/kernel/v/Read/ReadVariableOp)Adam/dense_637/bias/v/Read/ReadVariableOpConst*V
TinO
M2K	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__traced_save_15196585
Ê
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_627/kerneldense_627/biasdense_628/kerneldense_628/biasdense_629/kerneldense_629/biasdense_630/kerneldense_630/biasdense_631/kerneldense_631/biasdense_632/kerneldense_632/biasdense_633/kerneldense_633/biasdense_634/kerneldense_634/biasdense_635/kerneldense_635/biasdense_636/kerneldense_636/biasdense_637/kerneldense_637/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/dense_627/kernel/mAdam/dense_627/bias/mAdam/dense_628/kernel/mAdam/dense_628/bias/mAdam/dense_629/kernel/mAdam/dense_629/bias/mAdam/dense_630/kernel/mAdam/dense_630/bias/mAdam/dense_631/kernel/mAdam/dense_631/bias/mAdam/dense_632/kernel/mAdam/dense_632/bias/mAdam/dense_633/kernel/mAdam/dense_633/bias/mAdam/dense_634/kernel/mAdam/dense_634/bias/mAdam/dense_635/kernel/mAdam/dense_635/bias/mAdam/dense_636/kernel/mAdam/dense_636/bias/mAdam/dense_637/kernel/mAdam/dense_637/bias/mAdam/dense_627/kernel/vAdam/dense_627/bias/vAdam/dense_628/kernel/vAdam/dense_628/bias/vAdam/dense_629/kernel/vAdam/dense_629/bias/vAdam/dense_630/kernel/vAdam/dense_630/bias/vAdam/dense_631/kernel/vAdam/dense_631/bias/vAdam/dense_632/kernel/vAdam/dense_632/bias/vAdam/dense_633/kernel/vAdam/dense_633/bias/vAdam/dense_634/kernel/vAdam/dense_634/bias/vAdam/dense_635/kernel/vAdam/dense_635/bias/vAdam/dense_636/kernel/vAdam/dense_636/bias/vAdam/dense_637/kernel/vAdam/dense_637/bias/v*U
TinN
L2J*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference__traced_restore_15196814µõ



æ
G__inference_dense_633_layer_call_and_return_conditional_losses_15195407

inputs%
!mlcmatmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MLCMatMul/ReadVariableOp
MLCMatMul/ReadVariableOpReadVariableOp!mlcmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MLCMatMul/ReadVariableOp
	MLCMatMul	MLCMatMulinputs MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MLCMatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMLCMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
MLCMatMul/ReadVariableOpMLCMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¼	
æ
G__inference_dense_637_layer_call_and_return_conditional_losses_15196334

inputs%
!mlcmatmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MLCMatMul/ReadVariableOp
MLCMatMul/ReadVariableOpReadVariableOp!mlcmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MLCMatMul/ReadVariableOp
	MLCMatMul	MLCMatMulinputs MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MLCMatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMLCMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
MLCMatMul/ReadVariableOpMLCMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ã

,__inference_dense_634_layer_call_fn_15196284

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall÷
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_634_layer_call_and_return_conditional_losses_151954342
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


æ
G__inference_dense_632_layer_call_and_return_conditional_losses_15195380

inputs%
!mlcmatmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MLCMatMul/ReadVariableOp
MLCMatMul/ReadVariableOpReadVariableOp!mlcmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MLCMatMul/ReadVariableOp
	MLCMatMul	MLCMatMulinputs MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MLCMatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMLCMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
MLCMatMul/ReadVariableOpMLCMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
;

K__inference_sequential_57_layer_call_and_return_conditional_losses_15195590
dense_627_input
dense_627_15195534
dense_627_15195536
dense_628_15195539
dense_628_15195541
dense_629_15195544
dense_629_15195546
dense_630_15195549
dense_630_15195551
dense_631_15195554
dense_631_15195556
dense_632_15195559
dense_632_15195561
dense_633_15195564
dense_633_15195566
dense_634_15195569
dense_634_15195571
dense_635_15195574
dense_635_15195576
dense_636_15195579
dense_636_15195581
dense_637_15195584
dense_637_15195586
identity¢!dense_627/StatefulPartitionedCall¢!dense_628/StatefulPartitionedCall¢!dense_629/StatefulPartitionedCall¢!dense_630/StatefulPartitionedCall¢!dense_631/StatefulPartitionedCall¢!dense_632/StatefulPartitionedCall¢!dense_633/StatefulPartitionedCall¢!dense_634/StatefulPartitionedCall¢!dense_635/StatefulPartitionedCall¢!dense_636/StatefulPartitionedCall¢!dense_637/StatefulPartitionedCall¨
!dense_627/StatefulPartitionedCallStatefulPartitionedCalldense_627_inputdense_627_15195534dense_627_15195536*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_627_layer_call_and_return_conditional_losses_151952452#
!dense_627/StatefulPartitionedCallÃ
!dense_628/StatefulPartitionedCallStatefulPartitionedCall*dense_627/StatefulPartitionedCall:output:0dense_628_15195539dense_628_15195541*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_628_layer_call_and_return_conditional_losses_151952722#
!dense_628/StatefulPartitionedCallÃ
!dense_629/StatefulPartitionedCallStatefulPartitionedCall*dense_628/StatefulPartitionedCall:output:0dense_629_15195544dense_629_15195546*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_629_layer_call_and_return_conditional_losses_151952992#
!dense_629/StatefulPartitionedCallÃ
!dense_630/StatefulPartitionedCallStatefulPartitionedCall*dense_629/StatefulPartitionedCall:output:0dense_630_15195549dense_630_15195551*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_630_layer_call_and_return_conditional_losses_151953262#
!dense_630/StatefulPartitionedCallÃ
!dense_631/StatefulPartitionedCallStatefulPartitionedCall*dense_630/StatefulPartitionedCall:output:0dense_631_15195554dense_631_15195556*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_631_layer_call_and_return_conditional_losses_151953532#
!dense_631/StatefulPartitionedCallÃ
!dense_632/StatefulPartitionedCallStatefulPartitionedCall*dense_631/StatefulPartitionedCall:output:0dense_632_15195559dense_632_15195561*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_632_layer_call_and_return_conditional_losses_151953802#
!dense_632/StatefulPartitionedCallÃ
!dense_633/StatefulPartitionedCallStatefulPartitionedCall*dense_632/StatefulPartitionedCall:output:0dense_633_15195564dense_633_15195566*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_633_layer_call_and_return_conditional_losses_151954072#
!dense_633/StatefulPartitionedCallÃ
!dense_634/StatefulPartitionedCallStatefulPartitionedCall*dense_633/StatefulPartitionedCall:output:0dense_634_15195569dense_634_15195571*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_634_layer_call_and_return_conditional_losses_151954342#
!dense_634/StatefulPartitionedCallÃ
!dense_635/StatefulPartitionedCallStatefulPartitionedCall*dense_634/StatefulPartitionedCall:output:0dense_635_15195574dense_635_15195576*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_635_layer_call_and_return_conditional_losses_151954612#
!dense_635/StatefulPartitionedCallÃ
!dense_636/StatefulPartitionedCallStatefulPartitionedCall*dense_635/StatefulPartitionedCall:output:0dense_636_15195579dense_636_15195581*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_636_layer_call_and_return_conditional_losses_151954882#
!dense_636/StatefulPartitionedCallÃ
!dense_637/StatefulPartitionedCallStatefulPartitionedCall*dense_636/StatefulPartitionedCall:output:0dense_637_15195584dense_637_15195586*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_637_layer_call_and_return_conditional_losses_151955142#
!dense_637/StatefulPartitionedCall
IdentityIdentity*dense_637/StatefulPartitionedCall:output:0"^dense_627/StatefulPartitionedCall"^dense_628/StatefulPartitionedCall"^dense_629/StatefulPartitionedCall"^dense_630/StatefulPartitionedCall"^dense_631/StatefulPartitionedCall"^dense_632/StatefulPartitionedCall"^dense_633/StatefulPartitionedCall"^dense_634/StatefulPartitionedCall"^dense_635/StatefulPartitionedCall"^dense_636/StatefulPartitionedCall"^dense_637/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::2F
!dense_627/StatefulPartitionedCall!dense_627/StatefulPartitionedCall2F
!dense_628/StatefulPartitionedCall!dense_628/StatefulPartitionedCall2F
!dense_629/StatefulPartitionedCall!dense_629/StatefulPartitionedCall2F
!dense_630/StatefulPartitionedCall!dense_630/StatefulPartitionedCall2F
!dense_631/StatefulPartitionedCall!dense_631/StatefulPartitionedCall2F
!dense_632/StatefulPartitionedCall!dense_632/StatefulPartitionedCall2F
!dense_633/StatefulPartitionedCall!dense_633/StatefulPartitionedCall2F
!dense_634/StatefulPartitionedCall!dense_634/StatefulPartitionedCall2F
!dense_635/StatefulPartitionedCall!dense_635/StatefulPartitionedCall2F
!dense_636/StatefulPartitionedCall!dense_636/StatefulPartitionedCall2F
!dense_637/StatefulPartitionedCall!dense_637/StatefulPartitionedCall:X T
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)
_user_specified_namedense_627_input
¼	
æ
G__inference_dense_637_layer_call_and_return_conditional_losses_15195514

inputs%
!mlcmatmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MLCMatMul/ReadVariableOp
MLCMatMul/ReadVariableOpReadVariableOp!mlcmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MLCMatMul/ReadVariableOp
	MLCMatMul	MLCMatMulinputs MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MLCMatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMLCMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
MLCMatMul/ReadVariableOpMLCMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


æ
G__inference_dense_627_layer_call_and_return_conditional_losses_15196135

inputs%
!mlcmatmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MLCMatMul/ReadVariableOp
MLCMatMul/ReadVariableOpReadVariableOp!mlcmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MLCMatMul/ReadVariableOp
	MLCMatMul	MLCMatMulinputs MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MLCMatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMLCMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
MLCMatMul/ReadVariableOpMLCMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ü:

K__inference_sequential_57_layer_call_and_return_conditional_losses_15195760

inputs
dense_627_15195704
dense_627_15195706
dense_628_15195709
dense_628_15195711
dense_629_15195714
dense_629_15195716
dense_630_15195719
dense_630_15195721
dense_631_15195724
dense_631_15195726
dense_632_15195729
dense_632_15195731
dense_633_15195734
dense_633_15195736
dense_634_15195739
dense_634_15195741
dense_635_15195744
dense_635_15195746
dense_636_15195749
dense_636_15195751
dense_637_15195754
dense_637_15195756
identity¢!dense_627/StatefulPartitionedCall¢!dense_628/StatefulPartitionedCall¢!dense_629/StatefulPartitionedCall¢!dense_630/StatefulPartitionedCall¢!dense_631/StatefulPartitionedCall¢!dense_632/StatefulPartitionedCall¢!dense_633/StatefulPartitionedCall¢!dense_634/StatefulPartitionedCall¢!dense_635/StatefulPartitionedCall¢!dense_636/StatefulPartitionedCall¢!dense_637/StatefulPartitionedCall
!dense_627/StatefulPartitionedCallStatefulPartitionedCallinputsdense_627_15195704dense_627_15195706*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_627_layer_call_and_return_conditional_losses_151952452#
!dense_627/StatefulPartitionedCallÃ
!dense_628/StatefulPartitionedCallStatefulPartitionedCall*dense_627/StatefulPartitionedCall:output:0dense_628_15195709dense_628_15195711*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_628_layer_call_and_return_conditional_losses_151952722#
!dense_628/StatefulPartitionedCallÃ
!dense_629/StatefulPartitionedCallStatefulPartitionedCall*dense_628/StatefulPartitionedCall:output:0dense_629_15195714dense_629_15195716*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_629_layer_call_and_return_conditional_losses_151952992#
!dense_629/StatefulPartitionedCallÃ
!dense_630/StatefulPartitionedCallStatefulPartitionedCall*dense_629/StatefulPartitionedCall:output:0dense_630_15195719dense_630_15195721*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_630_layer_call_and_return_conditional_losses_151953262#
!dense_630/StatefulPartitionedCallÃ
!dense_631/StatefulPartitionedCallStatefulPartitionedCall*dense_630/StatefulPartitionedCall:output:0dense_631_15195724dense_631_15195726*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_631_layer_call_and_return_conditional_losses_151953532#
!dense_631/StatefulPartitionedCallÃ
!dense_632/StatefulPartitionedCallStatefulPartitionedCall*dense_631/StatefulPartitionedCall:output:0dense_632_15195729dense_632_15195731*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_632_layer_call_and_return_conditional_losses_151953802#
!dense_632/StatefulPartitionedCallÃ
!dense_633/StatefulPartitionedCallStatefulPartitionedCall*dense_632/StatefulPartitionedCall:output:0dense_633_15195734dense_633_15195736*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_633_layer_call_and_return_conditional_losses_151954072#
!dense_633/StatefulPartitionedCallÃ
!dense_634/StatefulPartitionedCallStatefulPartitionedCall*dense_633/StatefulPartitionedCall:output:0dense_634_15195739dense_634_15195741*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_634_layer_call_and_return_conditional_losses_151954342#
!dense_634/StatefulPartitionedCallÃ
!dense_635/StatefulPartitionedCallStatefulPartitionedCall*dense_634/StatefulPartitionedCall:output:0dense_635_15195744dense_635_15195746*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_635_layer_call_and_return_conditional_losses_151954612#
!dense_635/StatefulPartitionedCallÃ
!dense_636/StatefulPartitionedCallStatefulPartitionedCall*dense_635/StatefulPartitionedCall:output:0dense_636_15195749dense_636_15195751*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_636_layer_call_and_return_conditional_losses_151954882#
!dense_636/StatefulPartitionedCallÃ
!dense_637/StatefulPartitionedCallStatefulPartitionedCall*dense_636/StatefulPartitionedCall:output:0dense_637_15195754dense_637_15195756*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_637_layer_call_and_return_conditional_losses_151955142#
!dense_637/StatefulPartitionedCall
IdentityIdentity*dense_637/StatefulPartitionedCall:output:0"^dense_627/StatefulPartitionedCall"^dense_628/StatefulPartitionedCall"^dense_629/StatefulPartitionedCall"^dense_630/StatefulPartitionedCall"^dense_631/StatefulPartitionedCall"^dense_632/StatefulPartitionedCall"^dense_633/StatefulPartitionedCall"^dense_634/StatefulPartitionedCall"^dense_635/StatefulPartitionedCall"^dense_636/StatefulPartitionedCall"^dense_637/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::2F
!dense_627/StatefulPartitionedCall!dense_627/StatefulPartitionedCall2F
!dense_628/StatefulPartitionedCall!dense_628/StatefulPartitionedCall2F
!dense_629/StatefulPartitionedCall!dense_629/StatefulPartitionedCall2F
!dense_630/StatefulPartitionedCall!dense_630/StatefulPartitionedCall2F
!dense_631/StatefulPartitionedCall!dense_631/StatefulPartitionedCall2F
!dense_632/StatefulPartitionedCall!dense_632/StatefulPartitionedCall2F
!dense_633/StatefulPartitionedCall!dense_633/StatefulPartitionedCall2F
!dense_634/StatefulPartitionedCall!dense_634/StatefulPartitionedCall2F
!dense_635/StatefulPartitionedCall!dense_635/StatefulPartitionedCall2F
!dense_636/StatefulPartitionedCall!dense_636/StatefulPartitionedCall2F
!dense_637/StatefulPartitionedCall!dense_637/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


æ
G__inference_dense_630_layer_call_and_return_conditional_losses_15196195

inputs%
!mlcmatmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MLCMatMul/ReadVariableOp
MLCMatMul/ReadVariableOpReadVariableOp!mlcmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MLCMatMul/ReadVariableOp
	MLCMatMul	MLCMatMulinputs MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MLCMatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMLCMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
MLCMatMul/ReadVariableOpMLCMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


æ
G__inference_dense_630_layer_call_and_return_conditional_losses_15195326

inputs%
!mlcmatmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MLCMatMul/ReadVariableOp
MLCMatMul/ReadVariableOpReadVariableOp!mlcmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MLCMatMul/ReadVariableOp
	MLCMatMul	MLCMatMulinputs MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MLCMatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMLCMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
MLCMatMul/ReadVariableOpMLCMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


æ
G__inference_dense_635_layer_call_and_return_conditional_losses_15195461

inputs%
!mlcmatmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MLCMatMul/ReadVariableOp
MLCMatMul/ReadVariableOpReadVariableOp!mlcmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MLCMatMul/ReadVariableOp
	MLCMatMul	MLCMatMulinputs MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MLCMatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMLCMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
MLCMatMul/ReadVariableOpMLCMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


æ
G__inference_dense_633_layer_call_and_return_conditional_losses_15196255

inputs%
!mlcmatmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MLCMatMul/ReadVariableOp
MLCMatMul/ReadVariableOpReadVariableOp!mlcmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MLCMatMul/ReadVariableOp
	MLCMatMul	MLCMatMulinputs MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MLCMatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMLCMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
MLCMatMul/ReadVariableOpMLCMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ü:

K__inference_sequential_57_layer_call_and_return_conditional_losses_15195652

inputs
dense_627_15195596
dense_627_15195598
dense_628_15195601
dense_628_15195603
dense_629_15195606
dense_629_15195608
dense_630_15195611
dense_630_15195613
dense_631_15195616
dense_631_15195618
dense_632_15195621
dense_632_15195623
dense_633_15195626
dense_633_15195628
dense_634_15195631
dense_634_15195633
dense_635_15195636
dense_635_15195638
dense_636_15195641
dense_636_15195643
dense_637_15195646
dense_637_15195648
identity¢!dense_627/StatefulPartitionedCall¢!dense_628/StatefulPartitionedCall¢!dense_629/StatefulPartitionedCall¢!dense_630/StatefulPartitionedCall¢!dense_631/StatefulPartitionedCall¢!dense_632/StatefulPartitionedCall¢!dense_633/StatefulPartitionedCall¢!dense_634/StatefulPartitionedCall¢!dense_635/StatefulPartitionedCall¢!dense_636/StatefulPartitionedCall¢!dense_637/StatefulPartitionedCall
!dense_627/StatefulPartitionedCallStatefulPartitionedCallinputsdense_627_15195596dense_627_15195598*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_627_layer_call_and_return_conditional_losses_151952452#
!dense_627/StatefulPartitionedCallÃ
!dense_628/StatefulPartitionedCallStatefulPartitionedCall*dense_627/StatefulPartitionedCall:output:0dense_628_15195601dense_628_15195603*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_628_layer_call_and_return_conditional_losses_151952722#
!dense_628/StatefulPartitionedCallÃ
!dense_629/StatefulPartitionedCallStatefulPartitionedCall*dense_628/StatefulPartitionedCall:output:0dense_629_15195606dense_629_15195608*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_629_layer_call_and_return_conditional_losses_151952992#
!dense_629/StatefulPartitionedCallÃ
!dense_630/StatefulPartitionedCallStatefulPartitionedCall*dense_629/StatefulPartitionedCall:output:0dense_630_15195611dense_630_15195613*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_630_layer_call_and_return_conditional_losses_151953262#
!dense_630/StatefulPartitionedCallÃ
!dense_631/StatefulPartitionedCallStatefulPartitionedCall*dense_630/StatefulPartitionedCall:output:0dense_631_15195616dense_631_15195618*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_631_layer_call_and_return_conditional_losses_151953532#
!dense_631/StatefulPartitionedCallÃ
!dense_632/StatefulPartitionedCallStatefulPartitionedCall*dense_631/StatefulPartitionedCall:output:0dense_632_15195621dense_632_15195623*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_632_layer_call_and_return_conditional_losses_151953802#
!dense_632/StatefulPartitionedCallÃ
!dense_633/StatefulPartitionedCallStatefulPartitionedCall*dense_632/StatefulPartitionedCall:output:0dense_633_15195626dense_633_15195628*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_633_layer_call_and_return_conditional_losses_151954072#
!dense_633/StatefulPartitionedCallÃ
!dense_634/StatefulPartitionedCallStatefulPartitionedCall*dense_633/StatefulPartitionedCall:output:0dense_634_15195631dense_634_15195633*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_634_layer_call_and_return_conditional_losses_151954342#
!dense_634/StatefulPartitionedCallÃ
!dense_635/StatefulPartitionedCallStatefulPartitionedCall*dense_634/StatefulPartitionedCall:output:0dense_635_15195636dense_635_15195638*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_635_layer_call_and_return_conditional_losses_151954612#
!dense_635/StatefulPartitionedCallÃ
!dense_636/StatefulPartitionedCallStatefulPartitionedCall*dense_635/StatefulPartitionedCall:output:0dense_636_15195641dense_636_15195643*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_636_layer_call_and_return_conditional_losses_151954882#
!dense_636/StatefulPartitionedCallÃ
!dense_637/StatefulPartitionedCallStatefulPartitionedCall*dense_636/StatefulPartitionedCall:output:0dense_637_15195646dense_637_15195648*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_637_layer_call_and_return_conditional_losses_151955142#
!dense_637/StatefulPartitionedCall
IdentityIdentity*dense_637/StatefulPartitionedCall:output:0"^dense_627/StatefulPartitionedCall"^dense_628/StatefulPartitionedCall"^dense_629/StatefulPartitionedCall"^dense_630/StatefulPartitionedCall"^dense_631/StatefulPartitionedCall"^dense_632/StatefulPartitionedCall"^dense_633/StatefulPartitionedCall"^dense_634/StatefulPartitionedCall"^dense_635/StatefulPartitionedCall"^dense_636/StatefulPartitionedCall"^dense_637/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::2F
!dense_627/StatefulPartitionedCall!dense_627/StatefulPartitionedCall2F
!dense_628/StatefulPartitionedCall!dense_628/StatefulPartitionedCall2F
!dense_629/StatefulPartitionedCall!dense_629/StatefulPartitionedCall2F
!dense_630/StatefulPartitionedCall!dense_630/StatefulPartitionedCall2F
!dense_631/StatefulPartitionedCall!dense_631/StatefulPartitionedCall2F
!dense_632/StatefulPartitionedCall!dense_632/StatefulPartitionedCall2F
!dense_633/StatefulPartitionedCall!dense_633/StatefulPartitionedCall2F
!dense_634/StatefulPartitionedCall!dense_634/StatefulPartitionedCall2F
!dense_635/StatefulPartitionedCall!dense_635/StatefulPartitionedCall2F
!dense_636/StatefulPartitionedCall!dense_636/StatefulPartitionedCall2F
!dense_637/StatefulPartitionedCall!dense_637/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ã

,__inference_dense_628_layer_call_fn_15196164

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall÷
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_628_layer_call_and_return_conditional_losses_151952722
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


æ
G__inference_dense_634_layer_call_and_return_conditional_losses_15195434

inputs%
!mlcmatmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MLCMatMul/ReadVariableOp
MLCMatMul/ReadVariableOpReadVariableOp!mlcmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MLCMatMul/ReadVariableOp
	MLCMatMul	MLCMatMulinputs MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MLCMatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMLCMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
MLCMatMul/ReadVariableOpMLCMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ã

,__inference_dense_629_layer_call_fn_15196184

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall÷
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_629_layer_call_and_return_conditional_losses_151952992
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


æ
G__inference_dense_627_layer_call_and_return_conditional_losses_15195245

inputs%
!mlcmatmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MLCMatMul/ReadVariableOp
MLCMatMul/ReadVariableOpReadVariableOp!mlcmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MLCMatMul/ReadVariableOp
	MLCMatMul	MLCMatMulinputs MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MLCMatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMLCMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
MLCMatMul/ReadVariableOpMLCMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ë²
º&
$__inference__traced_restore_15196814
file_prefix%
!assignvariableop_dense_627_kernel%
!assignvariableop_1_dense_627_bias'
#assignvariableop_2_dense_628_kernel%
!assignvariableop_3_dense_628_bias'
#assignvariableop_4_dense_629_kernel%
!assignvariableop_5_dense_629_bias'
#assignvariableop_6_dense_630_kernel%
!assignvariableop_7_dense_630_bias'
#assignvariableop_8_dense_631_kernel%
!assignvariableop_9_dense_631_bias(
$assignvariableop_10_dense_632_kernel&
"assignvariableop_11_dense_632_bias(
$assignvariableop_12_dense_633_kernel&
"assignvariableop_13_dense_633_bias(
$assignvariableop_14_dense_634_kernel&
"assignvariableop_15_dense_634_bias(
$assignvariableop_16_dense_635_kernel&
"assignvariableop_17_dense_635_bias(
$assignvariableop_18_dense_636_kernel&
"assignvariableop_19_dense_636_bias(
$assignvariableop_20_dense_637_kernel&
"assignvariableop_21_dense_637_bias!
assignvariableop_22_adam_iter#
assignvariableop_23_adam_beta_1#
assignvariableop_24_adam_beta_2"
assignvariableop_25_adam_decay*
&assignvariableop_26_adam_learning_rate
assignvariableop_27_total
assignvariableop_28_count/
+assignvariableop_29_adam_dense_627_kernel_m-
)assignvariableop_30_adam_dense_627_bias_m/
+assignvariableop_31_adam_dense_628_kernel_m-
)assignvariableop_32_adam_dense_628_bias_m/
+assignvariableop_33_adam_dense_629_kernel_m-
)assignvariableop_34_adam_dense_629_bias_m/
+assignvariableop_35_adam_dense_630_kernel_m-
)assignvariableop_36_adam_dense_630_bias_m/
+assignvariableop_37_adam_dense_631_kernel_m-
)assignvariableop_38_adam_dense_631_bias_m/
+assignvariableop_39_adam_dense_632_kernel_m-
)assignvariableop_40_adam_dense_632_bias_m/
+assignvariableop_41_adam_dense_633_kernel_m-
)assignvariableop_42_adam_dense_633_bias_m/
+assignvariableop_43_adam_dense_634_kernel_m-
)assignvariableop_44_adam_dense_634_bias_m/
+assignvariableop_45_adam_dense_635_kernel_m-
)assignvariableop_46_adam_dense_635_bias_m/
+assignvariableop_47_adam_dense_636_kernel_m-
)assignvariableop_48_adam_dense_636_bias_m/
+assignvariableop_49_adam_dense_637_kernel_m-
)assignvariableop_50_adam_dense_637_bias_m/
+assignvariableop_51_adam_dense_627_kernel_v-
)assignvariableop_52_adam_dense_627_bias_v/
+assignvariableop_53_adam_dense_628_kernel_v-
)assignvariableop_54_adam_dense_628_bias_v/
+assignvariableop_55_adam_dense_629_kernel_v-
)assignvariableop_56_adam_dense_629_bias_v/
+assignvariableop_57_adam_dense_630_kernel_v-
)assignvariableop_58_adam_dense_630_bias_v/
+assignvariableop_59_adam_dense_631_kernel_v-
)assignvariableop_60_adam_dense_631_bias_v/
+assignvariableop_61_adam_dense_632_kernel_v-
)assignvariableop_62_adam_dense_632_bias_v/
+assignvariableop_63_adam_dense_633_kernel_v-
)assignvariableop_64_adam_dense_633_bias_v/
+assignvariableop_65_adam_dense_634_kernel_v-
)assignvariableop_66_adam_dense_634_bias_v/
+assignvariableop_67_adam_dense_635_kernel_v-
)assignvariableop_68_adam_dense_635_bias_v/
+assignvariableop_69_adam_dense_636_kernel_v-
)assignvariableop_70_adam_dense_636_bias_v/
+assignvariableop_71_adam_dense_637_kernel_v-
)assignvariableop_72_adam_dense_637_bias_v
identity_74¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_52¢AssignVariableOp_53¢AssignVariableOp_54¢AssignVariableOp_55¢AssignVariableOp_56¢AssignVariableOp_57¢AssignVariableOp_58¢AssignVariableOp_59¢AssignVariableOp_6¢AssignVariableOp_60¢AssignVariableOp_61¢AssignVariableOp_62¢AssignVariableOp_63¢AssignVariableOp_64¢AssignVariableOp_65¢AssignVariableOp_66¢AssignVariableOp_67¢AssignVariableOp_68¢AssignVariableOp_69¢AssignVariableOp_7¢AssignVariableOp_70¢AssignVariableOp_71¢AssignVariableOp_72¢AssignVariableOp_8¢AssignVariableOp_9ü)
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:J*
dtype0*)
valueþ(Bû(JB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names¥
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:J*
dtype0*©
valueBJB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices 
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*¾
_output_shapes«
¨::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*X
dtypesN
L2J	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity 
AssignVariableOpAssignVariableOp!assignvariableop_dense_627_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¦
AssignVariableOp_1AssignVariableOp!assignvariableop_1_dense_627_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2¨
AssignVariableOp_2AssignVariableOp#assignvariableop_2_dense_628_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¦
AssignVariableOp_3AssignVariableOp!assignvariableop_3_dense_628_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4¨
AssignVariableOp_4AssignVariableOp#assignvariableop_4_dense_629_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¦
AssignVariableOp_5AssignVariableOp!assignvariableop_5_dense_629_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6¨
AssignVariableOp_6AssignVariableOp#assignvariableop_6_dense_630_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7¦
AssignVariableOp_7AssignVariableOp!assignvariableop_7_dense_630_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8¨
AssignVariableOp_8AssignVariableOp#assignvariableop_8_dense_631_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9¦
AssignVariableOp_9AssignVariableOp!assignvariableop_9_dense_631_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10¬
AssignVariableOp_10AssignVariableOp$assignvariableop_10_dense_632_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11ª
AssignVariableOp_11AssignVariableOp"assignvariableop_11_dense_632_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12¬
AssignVariableOp_12AssignVariableOp$assignvariableop_12_dense_633_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13ª
AssignVariableOp_13AssignVariableOp"assignvariableop_13_dense_633_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14¬
AssignVariableOp_14AssignVariableOp$assignvariableop_14_dense_634_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15ª
AssignVariableOp_15AssignVariableOp"assignvariableop_15_dense_634_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16¬
AssignVariableOp_16AssignVariableOp$assignvariableop_16_dense_635_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17ª
AssignVariableOp_17AssignVariableOp"assignvariableop_17_dense_635_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18¬
AssignVariableOp_18AssignVariableOp$assignvariableop_18_dense_636_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19ª
AssignVariableOp_19AssignVariableOp"assignvariableop_19_dense_636_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20¬
AssignVariableOp_20AssignVariableOp$assignvariableop_20_dense_637_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21ª
AssignVariableOp_21AssignVariableOp"assignvariableop_21_dense_637_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_22¥
AssignVariableOp_22AssignVariableOpassignvariableop_22_adam_iterIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23§
AssignVariableOp_23AssignVariableOpassignvariableop_23_adam_beta_1Identity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24§
AssignVariableOp_24AssignVariableOpassignvariableop_24_adam_beta_2Identity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25¦
AssignVariableOp_25AssignVariableOpassignvariableop_25_adam_decayIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26®
AssignVariableOp_26AssignVariableOp&assignvariableop_26_adam_learning_rateIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27¡
AssignVariableOp_27AssignVariableOpassignvariableop_27_totalIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28¡
AssignVariableOp_28AssignVariableOpassignvariableop_28_countIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29³
AssignVariableOp_29AssignVariableOp+assignvariableop_29_adam_dense_627_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30±
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adam_dense_627_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31³
AssignVariableOp_31AssignVariableOp+assignvariableop_31_adam_dense_628_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32±
AssignVariableOp_32AssignVariableOp)assignvariableop_32_adam_dense_628_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33³
AssignVariableOp_33AssignVariableOp+assignvariableop_33_adam_dense_629_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34±
AssignVariableOp_34AssignVariableOp)assignvariableop_34_adam_dense_629_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35³
AssignVariableOp_35AssignVariableOp+assignvariableop_35_adam_dense_630_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36±
AssignVariableOp_36AssignVariableOp)assignvariableop_36_adam_dense_630_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37³
AssignVariableOp_37AssignVariableOp+assignvariableop_37_adam_dense_631_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38±
AssignVariableOp_38AssignVariableOp)assignvariableop_38_adam_dense_631_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39³
AssignVariableOp_39AssignVariableOp+assignvariableop_39_adam_dense_632_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40±
AssignVariableOp_40AssignVariableOp)assignvariableop_40_adam_dense_632_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41³
AssignVariableOp_41AssignVariableOp+assignvariableop_41_adam_dense_633_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42±
AssignVariableOp_42AssignVariableOp)assignvariableop_42_adam_dense_633_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43³
AssignVariableOp_43AssignVariableOp+assignvariableop_43_adam_dense_634_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44±
AssignVariableOp_44AssignVariableOp)assignvariableop_44_adam_dense_634_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45³
AssignVariableOp_45AssignVariableOp+assignvariableop_45_adam_dense_635_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46±
AssignVariableOp_46AssignVariableOp)assignvariableop_46_adam_dense_635_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47³
AssignVariableOp_47AssignVariableOp+assignvariableop_47_adam_dense_636_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48±
AssignVariableOp_48AssignVariableOp)assignvariableop_48_adam_dense_636_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49³
AssignVariableOp_49AssignVariableOp+assignvariableop_49_adam_dense_637_kernel_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50±
AssignVariableOp_50AssignVariableOp)assignvariableop_50_adam_dense_637_bias_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51³
AssignVariableOp_51AssignVariableOp+assignvariableop_51_adam_dense_627_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52±
AssignVariableOp_52AssignVariableOp)assignvariableop_52_adam_dense_627_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53³
AssignVariableOp_53AssignVariableOp+assignvariableop_53_adam_dense_628_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54±
AssignVariableOp_54AssignVariableOp)assignvariableop_54_adam_dense_628_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55³
AssignVariableOp_55AssignVariableOp+assignvariableop_55_adam_dense_629_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56±
AssignVariableOp_56AssignVariableOp)assignvariableop_56_adam_dense_629_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57³
AssignVariableOp_57AssignVariableOp+assignvariableop_57_adam_dense_630_kernel_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58±
AssignVariableOp_58AssignVariableOp)assignvariableop_58_adam_dense_630_bias_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59³
AssignVariableOp_59AssignVariableOp+assignvariableop_59_adam_dense_631_kernel_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60±
AssignVariableOp_60AssignVariableOp)assignvariableop_60_adam_dense_631_bias_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61³
AssignVariableOp_61AssignVariableOp+assignvariableop_61_adam_dense_632_kernel_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62±
AssignVariableOp_62AssignVariableOp)assignvariableop_62_adam_dense_632_bias_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63³
AssignVariableOp_63AssignVariableOp+assignvariableop_63_adam_dense_633_kernel_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64±
AssignVariableOp_64AssignVariableOp)assignvariableop_64_adam_dense_633_bias_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65³
AssignVariableOp_65AssignVariableOp+assignvariableop_65_adam_dense_634_kernel_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66±
AssignVariableOp_66AssignVariableOp)assignvariableop_66_adam_dense_634_bias_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67³
AssignVariableOp_67AssignVariableOp+assignvariableop_67_adam_dense_635_kernel_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68±
AssignVariableOp_68AssignVariableOp)assignvariableop_68_adam_dense_635_bias_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69³
AssignVariableOp_69AssignVariableOp+assignvariableop_69_adam_dense_636_kernel_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70±
AssignVariableOp_70AssignVariableOp)assignvariableop_70_adam_dense_636_bias_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71³
AssignVariableOp_71AssignVariableOp+assignvariableop_71_adam_dense_637_kernel_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72±
AssignVariableOp_72AssignVariableOp)assignvariableop_72_adam_dense_637_bias_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_729
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp¤
Identity_73Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_73
Identity_74IdentityIdentity_73:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_74"#
identity_74Identity_74:output:0*»
_input_shapes©
¦: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix


æ
G__inference_dense_629_layer_call_and_return_conditional_losses_15196175

inputs%
!mlcmatmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MLCMatMul/ReadVariableOp
MLCMatMul/ReadVariableOpReadVariableOp!mlcmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MLCMatMul/ReadVariableOp
	MLCMatMul	MLCMatMulinputs MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MLCMatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMLCMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
MLCMatMul/ReadVariableOpMLCMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ã

,__inference_dense_635_layer_call_fn_15196304

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall÷
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_635_layer_call_and_return_conditional_losses_151954612
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ã

,__inference_dense_636_layer_call_fn_15196324

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall÷
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_636_layer_call_and_return_conditional_losses_151954882
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
k
¢
K__inference_sequential_57_layer_call_and_return_conditional_losses_15195946

inputs/
+dense_627_mlcmatmul_readvariableop_resource-
)dense_627_biasadd_readvariableop_resource/
+dense_628_mlcmatmul_readvariableop_resource-
)dense_628_biasadd_readvariableop_resource/
+dense_629_mlcmatmul_readvariableop_resource-
)dense_629_biasadd_readvariableop_resource/
+dense_630_mlcmatmul_readvariableop_resource-
)dense_630_biasadd_readvariableop_resource/
+dense_631_mlcmatmul_readvariableop_resource-
)dense_631_biasadd_readvariableop_resource/
+dense_632_mlcmatmul_readvariableop_resource-
)dense_632_biasadd_readvariableop_resource/
+dense_633_mlcmatmul_readvariableop_resource-
)dense_633_biasadd_readvariableop_resource/
+dense_634_mlcmatmul_readvariableop_resource-
)dense_634_biasadd_readvariableop_resource/
+dense_635_mlcmatmul_readvariableop_resource-
)dense_635_biasadd_readvariableop_resource/
+dense_636_mlcmatmul_readvariableop_resource-
)dense_636_biasadd_readvariableop_resource/
+dense_637_mlcmatmul_readvariableop_resource-
)dense_637_biasadd_readvariableop_resource
identity¢ dense_627/BiasAdd/ReadVariableOp¢"dense_627/MLCMatMul/ReadVariableOp¢ dense_628/BiasAdd/ReadVariableOp¢"dense_628/MLCMatMul/ReadVariableOp¢ dense_629/BiasAdd/ReadVariableOp¢"dense_629/MLCMatMul/ReadVariableOp¢ dense_630/BiasAdd/ReadVariableOp¢"dense_630/MLCMatMul/ReadVariableOp¢ dense_631/BiasAdd/ReadVariableOp¢"dense_631/MLCMatMul/ReadVariableOp¢ dense_632/BiasAdd/ReadVariableOp¢"dense_632/MLCMatMul/ReadVariableOp¢ dense_633/BiasAdd/ReadVariableOp¢"dense_633/MLCMatMul/ReadVariableOp¢ dense_634/BiasAdd/ReadVariableOp¢"dense_634/MLCMatMul/ReadVariableOp¢ dense_635/BiasAdd/ReadVariableOp¢"dense_635/MLCMatMul/ReadVariableOp¢ dense_636/BiasAdd/ReadVariableOp¢"dense_636/MLCMatMul/ReadVariableOp¢ dense_637/BiasAdd/ReadVariableOp¢"dense_637/MLCMatMul/ReadVariableOp´
"dense_627/MLCMatMul/ReadVariableOpReadVariableOp+dense_627_mlcmatmul_readvariableop_resource*
_output_shapes

:*
dtype02$
"dense_627/MLCMatMul/ReadVariableOp
dense_627/MLCMatMul	MLCMatMulinputs*dense_627/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_627/MLCMatMulª
 dense_627/BiasAdd/ReadVariableOpReadVariableOp)dense_627_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_627/BiasAdd/ReadVariableOp¬
dense_627/BiasAddBiasAdddense_627/MLCMatMul:product:0(dense_627/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_627/BiasAddv
dense_627/ReluReludense_627/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_627/Relu´
"dense_628/MLCMatMul/ReadVariableOpReadVariableOp+dense_628_mlcmatmul_readvariableop_resource*
_output_shapes

:*
dtype02$
"dense_628/MLCMatMul/ReadVariableOp³
dense_628/MLCMatMul	MLCMatMuldense_627/Relu:activations:0*dense_628/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_628/MLCMatMulª
 dense_628/BiasAdd/ReadVariableOpReadVariableOp)dense_628_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_628/BiasAdd/ReadVariableOp¬
dense_628/BiasAddBiasAdddense_628/MLCMatMul:product:0(dense_628/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_628/BiasAddv
dense_628/ReluReludense_628/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_628/Relu´
"dense_629/MLCMatMul/ReadVariableOpReadVariableOp+dense_629_mlcmatmul_readvariableop_resource*
_output_shapes

:*
dtype02$
"dense_629/MLCMatMul/ReadVariableOp³
dense_629/MLCMatMul	MLCMatMuldense_628/Relu:activations:0*dense_629/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_629/MLCMatMulª
 dense_629/BiasAdd/ReadVariableOpReadVariableOp)dense_629_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_629/BiasAdd/ReadVariableOp¬
dense_629/BiasAddBiasAdddense_629/MLCMatMul:product:0(dense_629/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_629/BiasAddv
dense_629/ReluReludense_629/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_629/Relu´
"dense_630/MLCMatMul/ReadVariableOpReadVariableOp+dense_630_mlcmatmul_readvariableop_resource*
_output_shapes

:*
dtype02$
"dense_630/MLCMatMul/ReadVariableOp³
dense_630/MLCMatMul	MLCMatMuldense_629/Relu:activations:0*dense_630/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_630/MLCMatMulª
 dense_630/BiasAdd/ReadVariableOpReadVariableOp)dense_630_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_630/BiasAdd/ReadVariableOp¬
dense_630/BiasAddBiasAdddense_630/MLCMatMul:product:0(dense_630/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_630/BiasAddv
dense_630/ReluReludense_630/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_630/Relu´
"dense_631/MLCMatMul/ReadVariableOpReadVariableOp+dense_631_mlcmatmul_readvariableop_resource*
_output_shapes

:*
dtype02$
"dense_631/MLCMatMul/ReadVariableOp³
dense_631/MLCMatMul	MLCMatMuldense_630/Relu:activations:0*dense_631/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_631/MLCMatMulª
 dense_631/BiasAdd/ReadVariableOpReadVariableOp)dense_631_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_631/BiasAdd/ReadVariableOp¬
dense_631/BiasAddBiasAdddense_631/MLCMatMul:product:0(dense_631/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_631/BiasAddv
dense_631/ReluReludense_631/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_631/Relu´
"dense_632/MLCMatMul/ReadVariableOpReadVariableOp+dense_632_mlcmatmul_readvariableop_resource*
_output_shapes

:*
dtype02$
"dense_632/MLCMatMul/ReadVariableOp³
dense_632/MLCMatMul	MLCMatMuldense_631/Relu:activations:0*dense_632/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_632/MLCMatMulª
 dense_632/BiasAdd/ReadVariableOpReadVariableOp)dense_632_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_632/BiasAdd/ReadVariableOp¬
dense_632/BiasAddBiasAdddense_632/MLCMatMul:product:0(dense_632/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_632/BiasAddv
dense_632/ReluReludense_632/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_632/Relu´
"dense_633/MLCMatMul/ReadVariableOpReadVariableOp+dense_633_mlcmatmul_readvariableop_resource*
_output_shapes

:*
dtype02$
"dense_633/MLCMatMul/ReadVariableOp³
dense_633/MLCMatMul	MLCMatMuldense_632/Relu:activations:0*dense_633/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_633/MLCMatMulª
 dense_633/BiasAdd/ReadVariableOpReadVariableOp)dense_633_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_633/BiasAdd/ReadVariableOp¬
dense_633/BiasAddBiasAdddense_633/MLCMatMul:product:0(dense_633/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_633/BiasAddv
dense_633/ReluReludense_633/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_633/Relu´
"dense_634/MLCMatMul/ReadVariableOpReadVariableOp+dense_634_mlcmatmul_readvariableop_resource*
_output_shapes

:*
dtype02$
"dense_634/MLCMatMul/ReadVariableOp³
dense_634/MLCMatMul	MLCMatMuldense_633/Relu:activations:0*dense_634/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_634/MLCMatMulª
 dense_634/BiasAdd/ReadVariableOpReadVariableOp)dense_634_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_634/BiasAdd/ReadVariableOp¬
dense_634/BiasAddBiasAdddense_634/MLCMatMul:product:0(dense_634/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_634/BiasAddv
dense_634/ReluReludense_634/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_634/Relu´
"dense_635/MLCMatMul/ReadVariableOpReadVariableOp+dense_635_mlcmatmul_readvariableop_resource*
_output_shapes

:*
dtype02$
"dense_635/MLCMatMul/ReadVariableOp³
dense_635/MLCMatMul	MLCMatMuldense_634/Relu:activations:0*dense_635/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_635/MLCMatMulª
 dense_635/BiasAdd/ReadVariableOpReadVariableOp)dense_635_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_635/BiasAdd/ReadVariableOp¬
dense_635/BiasAddBiasAdddense_635/MLCMatMul:product:0(dense_635/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_635/BiasAddv
dense_635/ReluReludense_635/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_635/Relu´
"dense_636/MLCMatMul/ReadVariableOpReadVariableOp+dense_636_mlcmatmul_readvariableop_resource*
_output_shapes

:*
dtype02$
"dense_636/MLCMatMul/ReadVariableOp³
dense_636/MLCMatMul	MLCMatMuldense_635/Relu:activations:0*dense_636/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_636/MLCMatMulª
 dense_636/BiasAdd/ReadVariableOpReadVariableOp)dense_636_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_636/BiasAdd/ReadVariableOp¬
dense_636/BiasAddBiasAdddense_636/MLCMatMul:product:0(dense_636/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_636/BiasAddv
dense_636/ReluReludense_636/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_636/Relu´
"dense_637/MLCMatMul/ReadVariableOpReadVariableOp+dense_637_mlcmatmul_readvariableop_resource*
_output_shapes

:*
dtype02$
"dense_637/MLCMatMul/ReadVariableOp³
dense_637/MLCMatMul	MLCMatMuldense_636/Relu:activations:0*dense_637/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_637/MLCMatMulª
 dense_637/BiasAdd/ReadVariableOpReadVariableOp)dense_637_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_637/BiasAdd/ReadVariableOp¬
dense_637/BiasAddBiasAdddense_637/MLCMatMul:product:0(dense_637/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_637/BiasAdd
IdentityIdentitydense_637/BiasAdd:output:0!^dense_627/BiasAdd/ReadVariableOp#^dense_627/MLCMatMul/ReadVariableOp!^dense_628/BiasAdd/ReadVariableOp#^dense_628/MLCMatMul/ReadVariableOp!^dense_629/BiasAdd/ReadVariableOp#^dense_629/MLCMatMul/ReadVariableOp!^dense_630/BiasAdd/ReadVariableOp#^dense_630/MLCMatMul/ReadVariableOp!^dense_631/BiasAdd/ReadVariableOp#^dense_631/MLCMatMul/ReadVariableOp!^dense_632/BiasAdd/ReadVariableOp#^dense_632/MLCMatMul/ReadVariableOp!^dense_633/BiasAdd/ReadVariableOp#^dense_633/MLCMatMul/ReadVariableOp!^dense_634/BiasAdd/ReadVariableOp#^dense_634/MLCMatMul/ReadVariableOp!^dense_635/BiasAdd/ReadVariableOp#^dense_635/MLCMatMul/ReadVariableOp!^dense_636/BiasAdd/ReadVariableOp#^dense_636/MLCMatMul/ReadVariableOp!^dense_637/BiasAdd/ReadVariableOp#^dense_637/MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::2D
 dense_627/BiasAdd/ReadVariableOp dense_627/BiasAdd/ReadVariableOp2H
"dense_627/MLCMatMul/ReadVariableOp"dense_627/MLCMatMul/ReadVariableOp2D
 dense_628/BiasAdd/ReadVariableOp dense_628/BiasAdd/ReadVariableOp2H
"dense_628/MLCMatMul/ReadVariableOp"dense_628/MLCMatMul/ReadVariableOp2D
 dense_629/BiasAdd/ReadVariableOp dense_629/BiasAdd/ReadVariableOp2H
"dense_629/MLCMatMul/ReadVariableOp"dense_629/MLCMatMul/ReadVariableOp2D
 dense_630/BiasAdd/ReadVariableOp dense_630/BiasAdd/ReadVariableOp2H
"dense_630/MLCMatMul/ReadVariableOp"dense_630/MLCMatMul/ReadVariableOp2D
 dense_631/BiasAdd/ReadVariableOp dense_631/BiasAdd/ReadVariableOp2H
"dense_631/MLCMatMul/ReadVariableOp"dense_631/MLCMatMul/ReadVariableOp2D
 dense_632/BiasAdd/ReadVariableOp dense_632/BiasAdd/ReadVariableOp2H
"dense_632/MLCMatMul/ReadVariableOp"dense_632/MLCMatMul/ReadVariableOp2D
 dense_633/BiasAdd/ReadVariableOp dense_633/BiasAdd/ReadVariableOp2H
"dense_633/MLCMatMul/ReadVariableOp"dense_633/MLCMatMul/ReadVariableOp2D
 dense_634/BiasAdd/ReadVariableOp dense_634/BiasAdd/ReadVariableOp2H
"dense_634/MLCMatMul/ReadVariableOp"dense_634/MLCMatMul/ReadVariableOp2D
 dense_635/BiasAdd/ReadVariableOp dense_635/BiasAdd/ReadVariableOp2H
"dense_635/MLCMatMul/ReadVariableOp"dense_635/MLCMatMul/ReadVariableOp2D
 dense_636/BiasAdd/ReadVariableOp dense_636/BiasAdd/ReadVariableOp2H
"dense_636/MLCMatMul/ReadVariableOp"dense_636/MLCMatMul/ReadVariableOp2D
 dense_637/BiasAdd/ReadVariableOp dense_637/BiasAdd/ReadVariableOp2H
"dense_637/MLCMatMul/ReadVariableOp"dense_637/MLCMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

ë
#__inference__wrapped_model_15195230
dense_627_input=
9sequential_57_dense_627_mlcmatmul_readvariableop_resource;
7sequential_57_dense_627_biasadd_readvariableop_resource=
9sequential_57_dense_628_mlcmatmul_readvariableop_resource;
7sequential_57_dense_628_biasadd_readvariableop_resource=
9sequential_57_dense_629_mlcmatmul_readvariableop_resource;
7sequential_57_dense_629_biasadd_readvariableop_resource=
9sequential_57_dense_630_mlcmatmul_readvariableop_resource;
7sequential_57_dense_630_biasadd_readvariableop_resource=
9sequential_57_dense_631_mlcmatmul_readvariableop_resource;
7sequential_57_dense_631_biasadd_readvariableop_resource=
9sequential_57_dense_632_mlcmatmul_readvariableop_resource;
7sequential_57_dense_632_biasadd_readvariableop_resource=
9sequential_57_dense_633_mlcmatmul_readvariableop_resource;
7sequential_57_dense_633_biasadd_readvariableop_resource=
9sequential_57_dense_634_mlcmatmul_readvariableop_resource;
7sequential_57_dense_634_biasadd_readvariableop_resource=
9sequential_57_dense_635_mlcmatmul_readvariableop_resource;
7sequential_57_dense_635_biasadd_readvariableop_resource=
9sequential_57_dense_636_mlcmatmul_readvariableop_resource;
7sequential_57_dense_636_biasadd_readvariableop_resource=
9sequential_57_dense_637_mlcmatmul_readvariableop_resource;
7sequential_57_dense_637_biasadd_readvariableop_resource
identity¢.sequential_57/dense_627/BiasAdd/ReadVariableOp¢0sequential_57/dense_627/MLCMatMul/ReadVariableOp¢.sequential_57/dense_628/BiasAdd/ReadVariableOp¢0sequential_57/dense_628/MLCMatMul/ReadVariableOp¢.sequential_57/dense_629/BiasAdd/ReadVariableOp¢0sequential_57/dense_629/MLCMatMul/ReadVariableOp¢.sequential_57/dense_630/BiasAdd/ReadVariableOp¢0sequential_57/dense_630/MLCMatMul/ReadVariableOp¢.sequential_57/dense_631/BiasAdd/ReadVariableOp¢0sequential_57/dense_631/MLCMatMul/ReadVariableOp¢.sequential_57/dense_632/BiasAdd/ReadVariableOp¢0sequential_57/dense_632/MLCMatMul/ReadVariableOp¢.sequential_57/dense_633/BiasAdd/ReadVariableOp¢0sequential_57/dense_633/MLCMatMul/ReadVariableOp¢.sequential_57/dense_634/BiasAdd/ReadVariableOp¢0sequential_57/dense_634/MLCMatMul/ReadVariableOp¢.sequential_57/dense_635/BiasAdd/ReadVariableOp¢0sequential_57/dense_635/MLCMatMul/ReadVariableOp¢.sequential_57/dense_636/BiasAdd/ReadVariableOp¢0sequential_57/dense_636/MLCMatMul/ReadVariableOp¢.sequential_57/dense_637/BiasAdd/ReadVariableOp¢0sequential_57/dense_637/MLCMatMul/ReadVariableOpÞ
0sequential_57/dense_627/MLCMatMul/ReadVariableOpReadVariableOp9sequential_57_dense_627_mlcmatmul_readvariableop_resource*
_output_shapes

:*
dtype022
0sequential_57/dense_627/MLCMatMul/ReadVariableOpÐ
!sequential_57/dense_627/MLCMatMul	MLCMatMuldense_627_input8sequential_57/dense_627/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!sequential_57/dense_627/MLCMatMulÔ
.sequential_57/dense_627/BiasAdd/ReadVariableOpReadVariableOp7sequential_57_dense_627_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.sequential_57/dense_627/BiasAdd/ReadVariableOpä
sequential_57/dense_627/BiasAddBiasAdd+sequential_57/dense_627/MLCMatMul:product:06sequential_57/dense_627/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
sequential_57/dense_627/BiasAdd 
sequential_57/dense_627/ReluRelu(sequential_57/dense_627/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_57/dense_627/ReluÞ
0sequential_57/dense_628/MLCMatMul/ReadVariableOpReadVariableOp9sequential_57_dense_628_mlcmatmul_readvariableop_resource*
_output_shapes

:*
dtype022
0sequential_57/dense_628/MLCMatMul/ReadVariableOpë
!sequential_57/dense_628/MLCMatMul	MLCMatMul*sequential_57/dense_627/Relu:activations:08sequential_57/dense_628/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!sequential_57/dense_628/MLCMatMulÔ
.sequential_57/dense_628/BiasAdd/ReadVariableOpReadVariableOp7sequential_57_dense_628_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.sequential_57/dense_628/BiasAdd/ReadVariableOpä
sequential_57/dense_628/BiasAddBiasAdd+sequential_57/dense_628/MLCMatMul:product:06sequential_57/dense_628/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
sequential_57/dense_628/BiasAdd 
sequential_57/dense_628/ReluRelu(sequential_57/dense_628/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_57/dense_628/ReluÞ
0sequential_57/dense_629/MLCMatMul/ReadVariableOpReadVariableOp9sequential_57_dense_629_mlcmatmul_readvariableop_resource*
_output_shapes

:*
dtype022
0sequential_57/dense_629/MLCMatMul/ReadVariableOpë
!sequential_57/dense_629/MLCMatMul	MLCMatMul*sequential_57/dense_628/Relu:activations:08sequential_57/dense_629/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!sequential_57/dense_629/MLCMatMulÔ
.sequential_57/dense_629/BiasAdd/ReadVariableOpReadVariableOp7sequential_57_dense_629_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.sequential_57/dense_629/BiasAdd/ReadVariableOpä
sequential_57/dense_629/BiasAddBiasAdd+sequential_57/dense_629/MLCMatMul:product:06sequential_57/dense_629/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
sequential_57/dense_629/BiasAdd 
sequential_57/dense_629/ReluRelu(sequential_57/dense_629/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_57/dense_629/ReluÞ
0sequential_57/dense_630/MLCMatMul/ReadVariableOpReadVariableOp9sequential_57_dense_630_mlcmatmul_readvariableop_resource*
_output_shapes

:*
dtype022
0sequential_57/dense_630/MLCMatMul/ReadVariableOpë
!sequential_57/dense_630/MLCMatMul	MLCMatMul*sequential_57/dense_629/Relu:activations:08sequential_57/dense_630/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!sequential_57/dense_630/MLCMatMulÔ
.sequential_57/dense_630/BiasAdd/ReadVariableOpReadVariableOp7sequential_57_dense_630_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.sequential_57/dense_630/BiasAdd/ReadVariableOpä
sequential_57/dense_630/BiasAddBiasAdd+sequential_57/dense_630/MLCMatMul:product:06sequential_57/dense_630/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
sequential_57/dense_630/BiasAdd 
sequential_57/dense_630/ReluRelu(sequential_57/dense_630/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_57/dense_630/ReluÞ
0sequential_57/dense_631/MLCMatMul/ReadVariableOpReadVariableOp9sequential_57_dense_631_mlcmatmul_readvariableop_resource*
_output_shapes

:*
dtype022
0sequential_57/dense_631/MLCMatMul/ReadVariableOpë
!sequential_57/dense_631/MLCMatMul	MLCMatMul*sequential_57/dense_630/Relu:activations:08sequential_57/dense_631/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!sequential_57/dense_631/MLCMatMulÔ
.sequential_57/dense_631/BiasAdd/ReadVariableOpReadVariableOp7sequential_57_dense_631_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.sequential_57/dense_631/BiasAdd/ReadVariableOpä
sequential_57/dense_631/BiasAddBiasAdd+sequential_57/dense_631/MLCMatMul:product:06sequential_57/dense_631/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
sequential_57/dense_631/BiasAdd 
sequential_57/dense_631/ReluRelu(sequential_57/dense_631/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_57/dense_631/ReluÞ
0sequential_57/dense_632/MLCMatMul/ReadVariableOpReadVariableOp9sequential_57_dense_632_mlcmatmul_readvariableop_resource*
_output_shapes

:*
dtype022
0sequential_57/dense_632/MLCMatMul/ReadVariableOpë
!sequential_57/dense_632/MLCMatMul	MLCMatMul*sequential_57/dense_631/Relu:activations:08sequential_57/dense_632/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!sequential_57/dense_632/MLCMatMulÔ
.sequential_57/dense_632/BiasAdd/ReadVariableOpReadVariableOp7sequential_57_dense_632_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.sequential_57/dense_632/BiasAdd/ReadVariableOpä
sequential_57/dense_632/BiasAddBiasAdd+sequential_57/dense_632/MLCMatMul:product:06sequential_57/dense_632/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
sequential_57/dense_632/BiasAdd 
sequential_57/dense_632/ReluRelu(sequential_57/dense_632/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_57/dense_632/ReluÞ
0sequential_57/dense_633/MLCMatMul/ReadVariableOpReadVariableOp9sequential_57_dense_633_mlcmatmul_readvariableop_resource*
_output_shapes

:*
dtype022
0sequential_57/dense_633/MLCMatMul/ReadVariableOpë
!sequential_57/dense_633/MLCMatMul	MLCMatMul*sequential_57/dense_632/Relu:activations:08sequential_57/dense_633/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!sequential_57/dense_633/MLCMatMulÔ
.sequential_57/dense_633/BiasAdd/ReadVariableOpReadVariableOp7sequential_57_dense_633_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.sequential_57/dense_633/BiasAdd/ReadVariableOpä
sequential_57/dense_633/BiasAddBiasAdd+sequential_57/dense_633/MLCMatMul:product:06sequential_57/dense_633/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
sequential_57/dense_633/BiasAdd 
sequential_57/dense_633/ReluRelu(sequential_57/dense_633/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_57/dense_633/ReluÞ
0sequential_57/dense_634/MLCMatMul/ReadVariableOpReadVariableOp9sequential_57_dense_634_mlcmatmul_readvariableop_resource*
_output_shapes

:*
dtype022
0sequential_57/dense_634/MLCMatMul/ReadVariableOpë
!sequential_57/dense_634/MLCMatMul	MLCMatMul*sequential_57/dense_633/Relu:activations:08sequential_57/dense_634/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!sequential_57/dense_634/MLCMatMulÔ
.sequential_57/dense_634/BiasAdd/ReadVariableOpReadVariableOp7sequential_57_dense_634_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.sequential_57/dense_634/BiasAdd/ReadVariableOpä
sequential_57/dense_634/BiasAddBiasAdd+sequential_57/dense_634/MLCMatMul:product:06sequential_57/dense_634/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
sequential_57/dense_634/BiasAdd 
sequential_57/dense_634/ReluRelu(sequential_57/dense_634/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_57/dense_634/ReluÞ
0sequential_57/dense_635/MLCMatMul/ReadVariableOpReadVariableOp9sequential_57_dense_635_mlcmatmul_readvariableop_resource*
_output_shapes

:*
dtype022
0sequential_57/dense_635/MLCMatMul/ReadVariableOpë
!sequential_57/dense_635/MLCMatMul	MLCMatMul*sequential_57/dense_634/Relu:activations:08sequential_57/dense_635/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!sequential_57/dense_635/MLCMatMulÔ
.sequential_57/dense_635/BiasAdd/ReadVariableOpReadVariableOp7sequential_57_dense_635_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.sequential_57/dense_635/BiasAdd/ReadVariableOpä
sequential_57/dense_635/BiasAddBiasAdd+sequential_57/dense_635/MLCMatMul:product:06sequential_57/dense_635/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
sequential_57/dense_635/BiasAdd 
sequential_57/dense_635/ReluRelu(sequential_57/dense_635/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_57/dense_635/ReluÞ
0sequential_57/dense_636/MLCMatMul/ReadVariableOpReadVariableOp9sequential_57_dense_636_mlcmatmul_readvariableop_resource*
_output_shapes

:*
dtype022
0sequential_57/dense_636/MLCMatMul/ReadVariableOpë
!sequential_57/dense_636/MLCMatMul	MLCMatMul*sequential_57/dense_635/Relu:activations:08sequential_57/dense_636/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!sequential_57/dense_636/MLCMatMulÔ
.sequential_57/dense_636/BiasAdd/ReadVariableOpReadVariableOp7sequential_57_dense_636_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.sequential_57/dense_636/BiasAdd/ReadVariableOpä
sequential_57/dense_636/BiasAddBiasAdd+sequential_57/dense_636/MLCMatMul:product:06sequential_57/dense_636/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
sequential_57/dense_636/BiasAdd 
sequential_57/dense_636/ReluRelu(sequential_57/dense_636/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_57/dense_636/ReluÞ
0sequential_57/dense_637/MLCMatMul/ReadVariableOpReadVariableOp9sequential_57_dense_637_mlcmatmul_readvariableop_resource*
_output_shapes

:*
dtype022
0sequential_57/dense_637/MLCMatMul/ReadVariableOpë
!sequential_57/dense_637/MLCMatMul	MLCMatMul*sequential_57/dense_636/Relu:activations:08sequential_57/dense_637/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!sequential_57/dense_637/MLCMatMulÔ
.sequential_57/dense_637/BiasAdd/ReadVariableOpReadVariableOp7sequential_57_dense_637_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.sequential_57/dense_637/BiasAdd/ReadVariableOpä
sequential_57/dense_637/BiasAddBiasAdd+sequential_57/dense_637/MLCMatMul:product:06sequential_57/dense_637/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
sequential_57/dense_637/BiasAddÈ	
IdentityIdentity(sequential_57/dense_637/BiasAdd:output:0/^sequential_57/dense_627/BiasAdd/ReadVariableOp1^sequential_57/dense_627/MLCMatMul/ReadVariableOp/^sequential_57/dense_628/BiasAdd/ReadVariableOp1^sequential_57/dense_628/MLCMatMul/ReadVariableOp/^sequential_57/dense_629/BiasAdd/ReadVariableOp1^sequential_57/dense_629/MLCMatMul/ReadVariableOp/^sequential_57/dense_630/BiasAdd/ReadVariableOp1^sequential_57/dense_630/MLCMatMul/ReadVariableOp/^sequential_57/dense_631/BiasAdd/ReadVariableOp1^sequential_57/dense_631/MLCMatMul/ReadVariableOp/^sequential_57/dense_632/BiasAdd/ReadVariableOp1^sequential_57/dense_632/MLCMatMul/ReadVariableOp/^sequential_57/dense_633/BiasAdd/ReadVariableOp1^sequential_57/dense_633/MLCMatMul/ReadVariableOp/^sequential_57/dense_634/BiasAdd/ReadVariableOp1^sequential_57/dense_634/MLCMatMul/ReadVariableOp/^sequential_57/dense_635/BiasAdd/ReadVariableOp1^sequential_57/dense_635/MLCMatMul/ReadVariableOp/^sequential_57/dense_636/BiasAdd/ReadVariableOp1^sequential_57/dense_636/MLCMatMul/ReadVariableOp/^sequential_57/dense_637/BiasAdd/ReadVariableOp1^sequential_57/dense_637/MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::2`
.sequential_57/dense_627/BiasAdd/ReadVariableOp.sequential_57/dense_627/BiasAdd/ReadVariableOp2d
0sequential_57/dense_627/MLCMatMul/ReadVariableOp0sequential_57/dense_627/MLCMatMul/ReadVariableOp2`
.sequential_57/dense_628/BiasAdd/ReadVariableOp.sequential_57/dense_628/BiasAdd/ReadVariableOp2d
0sequential_57/dense_628/MLCMatMul/ReadVariableOp0sequential_57/dense_628/MLCMatMul/ReadVariableOp2`
.sequential_57/dense_629/BiasAdd/ReadVariableOp.sequential_57/dense_629/BiasAdd/ReadVariableOp2d
0sequential_57/dense_629/MLCMatMul/ReadVariableOp0sequential_57/dense_629/MLCMatMul/ReadVariableOp2`
.sequential_57/dense_630/BiasAdd/ReadVariableOp.sequential_57/dense_630/BiasAdd/ReadVariableOp2d
0sequential_57/dense_630/MLCMatMul/ReadVariableOp0sequential_57/dense_630/MLCMatMul/ReadVariableOp2`
.sequential_57/dense_631/BiasAdd/ReadVariableOp.sequential_57/dense_631/BiasAdd/ReadVariableOp2d
0sequential_57/dense_631/MLCMatMul/ReadVariableOp0sequential_57/dense_631/MLCMatMul/ReadVariableOp2`
.sequential_57/dense_632/BiasAdd/ReadVariableOp.sequential_57/dense_632/BiasAdd/ReadVariableOp2d
0sequential_57/dense_632/MLCMatMul/ReadVariableOp0sequential_57/dense_632/MLCMatMul/ReadVariableOp2`
.sequential_57/dense_633/BiasAdd/ReadVariableOp.sequential_57/dense_633/BiasAdd/ReadVariableOp2d
0sequential_57/dense_633/MLCMatMul/ReadVariableOp0sequential_57/dense_633/MLCMatMul/ReadVariableOp2`
.sequential_57/dense_634/BiasAdd/ReadVariableOp.sequential_57/dense_634/BiasAdd/ReadVariableOp2d
0sequential_57/dense_634/MLCMatMul/ReadVariableOp0sequential_57/dense_634/MLCMatMul/ReadVariableOp2`
.sequential_57/dense_635/BiasAdd/ReadVariableOp.sequential_57/dense_635/BiasAdd/ReadVariableOp2d
0sequential_57/dense_635/MLCMatMul/ReadVariableOp0sequential_57/dense_635/MLCMatMul/ReadVariableOp2`
.sequential_57/dense_636/BiasAdd/ReadVariableOp.sequential_57/dense_636/BiasAdd/ReadVariableOp2d
0sequential_57/dense_636/MLCMatMul/ReadVariableOp0sequential_57/dense_636/MLCMatMul/ReadVariableOp2`
.sequential_57/dense_637/BiasAdd/ReadVariableOp.sequential_57/dense_637/BiasAdd/ReadVariableOp2d
0sequential_57/dense_637/MLCMatMul/ReadVariableOp0sequential_57/dense_637/MLCMatMul/ReadVariableOp:X T
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)
_user_specified_namedense_627_input
ã

,__inference_dense_637_layer_call_fn_15196343

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall÷
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_637_layer_call_and_return_conditional_losses_151955142
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Å
0__inference_sequential_57_layer_call_fn_15195699
dense_627_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCalldense_627_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_sequential_57_layer_call_and_return_conditional_losses_151956522
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)
_user_specified_namedense_627_input


æ
G__inference_dense_631_layer_call_and_return_conditional_losses_15196215

inputs%
!mlcmatmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MLCMatMul/ReadVariableOp
MLCMatMul/ReadVariableOpReadVariableOp!mlcmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MLCMatMul/ReadVariableOp
	MLCMatMul	MLCMatMulinputs MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MLCMatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMLCMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
MLCMatMul/ReadVariableOpMLCMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

¼
0__inference_sequential_57_layer_call_fn_15196124

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_sequential_57_layer_call_and_return_conditional_losses_151957602
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


æ
G__inference_dense_634_layer_call_and_return_conditional_losses_15196275

inputs%
!mlcmatmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MLCMatMul/ReadVariableOp
MLCMatMul/ReadVariableOpReadVariableOp!mlcmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MLCMatMul/ReadVariableOp
	MLCMatMul	MLCMatMulinputs MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MLCMatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMLCMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
MLCMatMul/ReadVariableOpMLCMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


æ
G__inference_dense_632_layer_call_and_return_conditional_losses_15196235

inputs%
!mlcmatmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MLCMatMul/ReadVariableOp
MLCMatMul/ReadVariableOpReadVariableOp!mlcmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MLCMatMul/ReadVariableOp
	MLCMatMul	MLCMatMulinputs MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MLCMatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMLCMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
MLCMatMul/ReadVariableOpMLCMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


æ
G__inference_dense_636_layer_call_and_return_conditional_losses_15196315

inputs%
!mlcmatmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MLCMatMul/ReadVariableOp
MLCMatMul/ReadVariableOpReadVariableOp!mlcmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MLCMatMul/ReadVariableOp
	MLCMatMul	MLCMatMulinputs MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MLCMatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMLCMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
MLCMatMul/ReadVariableOpMLCMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


æ
G__inference_dense_631_layer_call_and_return_conditional_losses_15195353

inputs%
!mlcmatmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MLCMatMul/ReadVariableOp
MLCMatMul/ReadVariableOpReadVariableOp!mlcmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MLCMatMul/ReadVariableOp
	MLCMatMul	MLCMatMulinputs MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MLCMatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMLCMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
MLCMatMul/ReadVariableOpMLCMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ê
»
&__inference_signature_wrapper_15195866
dense_627_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identity¢StatefulPartitionedCallë
StatefulPartitionedCallStatefulPartitionedCalldense_627_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference__wrapped_model_151952302
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)
_user_specified_namedense_627_input
ã

,__inference_dense_630_layer_call_fn_15196204

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall÷
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_630_layer_call_and_return_conditional_losses_151953262
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


æ
G__inference_dense_629_layer_call_and_return_conditional_losses_15195299

inputs%
!mlcmatmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MLCMatMul/ReadVariableOp
MLCMatMul/ReadVariableOpReadVariableOp!mlcmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MLCMatMul/ReadVariableOp
	MLCMatMul	MLCMatMulinputs MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MLCMatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMLCMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
MLCMatMul/ReadVariableOpMLCMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
;

K__inference_sequential_57_layer_call_and_return_conditional_losses_15195531
dense_627_input
dense_627_15195256
dense_627_15195258
dense_628_15195283
dense_628_15195285
dense_629_15195310
dense_629_15195312
dense_630_15195337
dense_630_15195339
dense_631_15195364
dense_631_15195366
dense_632_15195391
dense_632_15195393
dense_633_15195418
dense_633_15195420
dense_634_15195445
dense_634_15195447
dense_635_15195472
dense_635_15195474
dense_636_15195499
dense_636_15195501
dense_637_15195525
dense_637_15195527
identity¢!dense_627/StatefulPartitionedCall¢!dense_628/StatefulPartitionedCall¢!dense_629/StatefulPartitionedCall¢!dense_630/StatefulPartitionedCall¢!dense_631/StatefulPartitionedCall¢!dense_632/StatefulPartitionedCall¢!dense_633/StatefulPartitionedCall¢!dense_634/StatefulPartitionedCall¢!dense_635/StatefulPartitionedCall¢!dense_636/StatefulPartitionedCall¢!dense_637/StatefulPartitionedCall¨
!dense_627/StatefulPartitionedCallStatefulPartitionedCalldense_627_inputdense_627_15195256dense_627_15195258*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_627_layer_call_and_return_conditional_losses_151952452#
!dense_627/StatefulPartitionedCallÃ
!dense_628/StatefulPartitionedCallStatefulPartitionedCall*dense_627/StatefulPartitionedCall:output:0dense_628_15195283dense_628_15195285*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_628_layer_call_and_return_conditional_losses_151952722#
!dense_628/StatefulPartitionedCallÃ
!dense_629/StatefulPartitionedCallStatefulPartitionedCall*dense_628/StatefulPartitionedCall:output:0dense_629_15195310dense_629_15195312*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_629_layer_call_and_return_conditional_losses_151952992#
!dense_629/StatefulPartitionedCallÃ
!dense_630/StatefulPartitionedCallStatefulPartitionedCall*dense_629/StatefulPartitionedCall:output:0dense_630_15195337dense_630_15195339*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_630_layer_call_and_return_conditional_losses_151953262#
!dense_630/StatefulPartitionedCallÃ
!dense_631/StatefulPartitionedCallStatefulPartitionedCall*dense_630/StatefulPartitionedCall:output:0dense_631_15195364dense_631_15195366*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_631_layer_call_and_return_conditional_losses_151953532#
!dense_631/StatefulPartitionedCallÃ
!dense_632/StatefulPartitionedCallStatefulPartitionedCall*dense_631/StatefulPartitionedCall:output:0dense_632_15195391dense_632_15195393*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_632_layer_call_and_return_conditional_losses_151953802#
!dense_632/StatefulPartitionedCallÃ
!dense_633/StatefulPartitionedCallStatefulPartitionedCall*dense_632/StatefulPartitionedCall:output:0dense_633_15195418dense_633_15195420*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_633_layer_call_and_return_conditional_losses_151954072#
!dense_633/StatefulPartitionedCallÃ
!dense_634/StatefulPartitionedCallStatefulPartitionedCall*dense_633/StatefulPartitionedCall:output:0dense_634_15195445dense_634_15195447*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_634_layer_call_and_return_conditional_losses_151954342#
!dense_634/StatefulPartitionedCallÃ
!dense_635/StatefulPartitionedCallStatefulPartitionedCall*dense_634/StatefulPartitionedCall:output:0dense_635_15195472dense_635_15195474*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_635_layer_call_and_return_conditional_losses_151954612#
!dense_635/StatefulPartitionedCallÃ
!dense_636/StatefulPartitionedCallStatefulPartitionedCall*dense_635/StatefulPartitionedCall:output:0dense_636_15195499dense_636_15195501*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_636_layer_call_and_return_conditional_losses_151954882#
!dense_636/StatefulPartitionedCallÃ
!dense_637/StatefulPartitionedCallStatefulPartitionedCall*dense_636/StatefulPartitionedCall:output:0dense_637_15195525dense_637_15195527*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_637_layer_call_and_return_conditional_losses_151955142#
!dense_637/StatefulPartitionedCall
IdentityIdentity*dense_637/StatefulPartitionedCall:output:0"^dense_627/StatefulPartitionedCall"^dense_628/StatefulPartitionedCall"^dense_629/StatefulPartitionedCall"^dense_630/StatefulPartitionedCall"^dense_631/StatefulPartitionedCall"^dense_632/StatefulPartitionedCall"^dense_633/StatefulPartitionedCall"^dense_634/StatefulPartitionedCall"^dense_635/StatefulPartitionedCall"^dense_636/StatefulPartitionedCall"^dense_637/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::2F
!dense_627/StatefulPartitionedCall!dense_627/StatefulPartitionedCall2F
!dense_628/StatefulPartitionedCall!dense_628/StatefulPartitionedCall2F
!dense_629/StatefulPartitionedCall!dense_629/StatefulPartitionedCall2F
!dense_630/StatefulPartitionedCall!dense_630/StatefulPartitionedCall2F
!dense_631/StatefulPartitionedCall!dense_631/StatefulPartitionedCall2F
!dense_632/StatefulPartitionedCall!dense_632/StatefulPartitionedCall2F
!dense_633/StatefulPartitionedCall!dense_633/StatefulPartitionedCall2F
!dense_634/StatefulPartitionedCall!dense_634/StatefulPartitionedCall2F
!dense_635/StatefulPartitionedCall!dense_635/StatefulPartitionedCall2F
!dense_636/StatefulPartitionedCall!dense_636/StatefulPartitionedCall2F
!dense_637/StatefulPartitionedCall!dense_637/StatefulPartitionedCall:X T
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)
_user_specified_namedense_627_input
ã

,__inference_dense_627_layer_call_fn_15196144

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall÷
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_627_layer_call_and_return_conditional_losses_151952452
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ã

,__inference_dense_632_layer_call_fn_15196244

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall÷
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_632_layer_call_and_return_conditional_losses_151953802
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ã

,__inference_dense_633_layer_call_fn_15196264

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall÷
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_633_layer_call_and_return_conditional_losses_151954072
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ã

,__inference_dense_631_layer_call_fn_15196224

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall÷
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dense_631_layer_call_and_return_conditional_losses_151953532
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


æ
G__inference_dense_628_layer_call_and_return_conditional_losses_15196155

inputs%
!mlcmatmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MLCMatMul/ReadVariableOp
MLCMatMul/ReadVariableOpReadVariableOp!mlcmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MLCMatMul/ReadVariableOp
	MLCMatMul	MLCMatMulinputs MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MLCMatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMLCMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
MLCMatMul/ReadVariableOpMLCMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

¼
0__inference_sequential_57_layer_call_fn_15196075

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_sequential_57_layer_call_and_return_conditional_losses_151956522
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


æ
G__inference_dense_628_layer_call_and_return_conditional_losses_15195272

inputs%
!mlcmatmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MLCMatMul/ReadVariableOp
MLCMatMul/ReadVariableOpReadVariableOp!mlcmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MLCMatMul/ReadVariableOp
	MLCMatMul	MLCMatMulinputs MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MLCMatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMLCMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
MLCMatMul/ReadVariableOpMLCMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¥
®
!__inference__traced_save_15196585
file_prefix/
+savev2_dense_627_kernel_read_readvariableop-
)savev2_dense_627_bias_read_readvariableop/
+savev2_dense_628_kernel_read_readvariableop-
)savev2_dense_628_bias_read_readvariableop/
+savev2_dense_629_kernel_read_readvariableop-
)savev2_dense_629_bias_read_readvariableop/
+savev2_dense_630_kernel_read_readvariableop-
)savev2_dense_630_bias_read_readvariableop/
+savev2_dense_631_kernel_read_readvariableop-
)savev2_dense_631_bias_read_readvariableop/
+savev2_dense_632_kernel_read_readvariableop-
)savev2_dense_632_bias_read_readvariableop/
+savev2_dense_633_kernel_read_readvariableop-
)savev2_dense_633_bias_read_readvariableop/
+savev2_dense_634_kernel_read_readvariableop-
)savev2_dense_634_bias_read_readvariableop/
+savev2_dense_635_kernel_read_readvariableop-
)savev2_dense_635_bias_read_readvariableop/
+savev2_dense_636_kernel_read_readvariableop-
)savev2_dense_636_bias_read_readvariableop/
+savev2_dense_637_kernel_read_readvariableop-
)savev2_dense_637_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop6
2savev2_adam_dense_627_kernel_m_read_readvariableop4
0savev2_adam_dense_627_bias_m_read_readvariableop6
2savev2_adam_dense_628_kernel_m_read_readvariableop4
0savev2_adam_dense_628_bias_m_read_readvariableop6
2savev2_adam_dense_629_kernel_m_read_readvariableop4
0savev2_adam_dense_629_bias_m_read_readvariableop6
2savev2_adam_dense_630_kernel_m_read_readvariableop4
0savev2_adam_dense_630_bias_m_read_readvariableop6
2savev2_adam_dense_631_kernel_m_read_readvariableop4
0savev2_adam_dense_631_bias_m_read_readvariableop6
2savev2_adam_dense_632_kernel_m_read_readvariableop4
0savev2_adam_dense_632_bias_m_read_readvariableop6
2savev2_adam_dense_633_kernel_m_read_readvariableop4
0savev2_adam_dense_633_bias_m_read_readvariableop6
2savev2_adam_dense_634_kernel_m_read_readvariableop4
0savev2_adam_dense_634_bias_m_read_readvariableop6
2savev2_adam_dense_635_kernel_m_read_readvariableop4
0savev2_adam_dense_635_bias_m_read_readvariableop6
2savev2_adam_dense_636_kernel_m_read_readvariableop4
0savev2_adam_dense_636_bias_m_read_readvariableop6
2savev2_adam_dense_637_kernel_m_read_readvariableop4
0savev2_adam_dense_637_bias_m_read_readvariableop6
2savev2_adam_dense_627_kernel_v_read_readvariableop4
0savev2_adam_dense_627_bias_v_read_readvariableop6
2savev2_adam_dense_628_kernel_v_read_readvariableop4
0savev2_adam_dense_628_bias_v_read_readvariableop6
2savev2_adam_dense_629_kernel_v_read_readvariableop4
0savev2_adam_dense_629_bias_v_read_readvariableop6
2savev2_adam_dense_630_kernel_v_read_readvariableop4
0savev2_adam_dense_630_bias_v_read_readvariableop6
2savev2_adam_dense_631_kernel_v_read_readvariableop4
0savev2_adam_dense_631_bias_v_read_readvariableop6
2savev2_adam_dense_632_kernel_v_read_readvariableop4
0savev2_adam_dense_632_bias_v_read_readvariableop6
2savev2_adam_dense_633_kernel_v_read_readvariableop4
0savev2_adam_dense_633_bias_v_read_readvariableop6
2savev2_adam_dense_634_kernel_v_read_readvariableop4
0savev2_adam_dense_634_bias_v_read_readvariableop6
2savev2_adam_dense_635_kernel_v_read_readvariableop4
0savev2_adam_dense_635_bias_v_read_readvariableop6
2savev2_adam_dense_636_kernel_v_read_readvariableop4
0savev2_adam_dense_636_bias_v_read_readvariableop6
2savev2_adam_dense_637_kernel_v_read_readvariableop4
0savev2_adam_dense_637_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameö)
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:J*
dtype0*)
valueþ(Bû(JB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:J*
dtype0*©
valueBJB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_dense_627_kernel_read_readvariableop)savev2_dense_627_bias_read_readvariableop+savev2_dense_628_kernel_read_readvariableop)savev2_dense_628_bias_read_readvariableop+savev2_dense_629_kernel_read_readvariableop)savev2_dense_629_bias_read_readvariableop+savev2_dense_630_kernel_read_readvariableop)savev2_dense_630_bias_read_readvariableop+savev2_dense_631_kernel_read_readvariableop)savev2_dense_631_bias_read_readvariableop+savev2_dense_632_kernel_read_readvariableop)savev2_dense_632_bias_read_readvariableop+savev2_dense_633_kernel_read_readvariableop)savev2_dense_633_bias_read_readvariableop+savev2_dense_634_kernel_read_readvariableop)savev2_dense_634_bias_read_readvariableop+savev2_dense_635_kernel_read_readvariableop)savev2_dense_635_bias_read_readvariableop+savev2_dense_636_kernel_read_readvariableop)savev2_dense_636_bias_read_readvariableop+savev2_dense_637_kernel_read_readvariableop)savev2_dense_637_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop2savev2_adam_dense_627_kernel_m_read_readvariableop0savev2_adam_dense_627_bias_m_read_readvariableop2savev2_adam_dense_628_kernel_m_read_readvariableop0savev2_adam_dense_628_bias_m_read_readvariableop2savev2_adam_dense_629_kernel_m_read_readvariableop0savev2_adam_dense_629_bias_m_read_readvariableop2savev2_adam_dense_630_kernel_m_read_readvariableop0savev2_adam_dense_630_bias_m_read_readvariableop2savev2_adam_dense_631_kernel_m_read_readvariableop0savev2_adam_dense_631_bias_m_read_readvariableop2savev2_adam_dense_632_kernel_m_read_readvariableop0savev2_adam_dense_632_bias_m_read_readvariableop2savev2_adam_dense_633_kernel_m_read_readvariableop0savev2_adam_dense_633_bias_m_read_readvariableop2savev2_adam_dense_634_kernel_m_read_readvariableop0savev2_adam_dense_634_bias_m_read_readvariableop2savev2_adam_dense_635_kernel_m_read_readvariableop0savev2_adam_dense_635_bias_m_read_readvariableop2savev2_adam_dense_636_kernel_m_read_readvariableop0savev2_adam_dense_636_bias_m_read_readvariableop2savev2_adam_dense_637_kernel_m_read_readvariableop0savev2_adam_dense_637_bias_m_read_readvariableop2savev2_adam_dense_627_kernel_v_read_readvariableop0savev2_adam_dense_627_bias_v_read_readvariableop2savev2_adam_dense_628_kernel_v_read_readvariableop0savev2_adam_dense_628_bias_v_read_readvariableop2savev2_adam_dense_629_kernel_v_read_readvariableop0savev2_adam_dense_629_bias_v_read_readvariableop2savev2_adam_dense_630_kernel_v_read_readvariableop0savev2_adam_dense_630_bias_v_read_readvariableop2savev2_adam_dense_631_kernel_v_read_readvariableop0savev2_adam_dense_631_bias_v_read_readvariableop2savev2_adam_dense_632_kernel_v_read_readvariableop0savev2_adam_dense_632_bias_v_read_readvariableop2savev2_adam_dense_633_kernel_v_read_readvariableop0savev2_adam_dense_633_bias_v_read_readvariableop2savev2_adam_dense_634_kernel_v_read_readvariableop0savev2_adam_dense_634_bias_v_read_readvariableop2savev2_adam_dense_635_kernel_v_read_readvariableop0savev2_adam_dense_635_bias_v_read_readvariableop2savev2_adam_dense_636_kernel_v_read_readvariableop0savev2_adam_dense_636_bias_v_read_readvariableop2savev2_adam_dense_637_kernel_v_read_readvariableop0savev2_adam_dense_637_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *X
dtypesN
L2J	2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*·
_input_shapes¥
¢: ::::::::::::::::::::::: : : : : : : ::::::::::::::::::::::::::::::::::::::::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$	 

_output_shapes

:: 


_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:: 

_output_shapes
::$  

_output_shapes

:: !

_output_shapes
::$" 

_output_shapes

:: #

_output_shapes
::$$ 

_output_shapes

:: %

_output_shapes
::$& 

_output_shapes

:: '

_output_shapes
::$( 

_output_shapes

:: )

_output_shapes
::$* 

_output_shapes

:: +

_output_shapes
::$, 

_output_shapes

:: -

_output_shapes
::$. 

_output_shapes

:: /

_output_shapes
::$0 

_output_shapes

:: 1

_output_shapes
::$2 

_output_shapes

:: 3

_output_shapes
::$4 

_output_shapes

:: 5

_output_shapes
::$6 

_output_shapes

:: 7

_output_shapes
::$8 

_output_shapes

:: 9

_output_shapes
::$: 

_output_shapes

:: ;

_output_shapes
::$< 

_output_shapes

:: =

_output_shapes
::$> 

_output_shapes

:: ?

_output_shapes
::$@ 

_output_shapes

:: A

_output_shapes
::$B 

_output_shapes

:: C

_output_shapes
::$D 

_output_shapes

:: E

_output_shapes
::$F 

_output_shapes

:: G

_output_shapes
::$H 

_output_shapes

:: I

_output_shapes
::J

_output_shapes
: 


æ
G__inference_dense_635_layer_call_and_return_conditional_losses_15196295

inputs%
!mlcmatmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MLCMatMul/ReadVariableOp
MLCMatMul/ReadVariableOpReadVariableOp!mlcmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MLCMatMul/ReadVariableOp
	MLCMatMul	MLCMatMulinputs MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MLCMatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMLCMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
MLCMatMul/ReadVariableOpMLCMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
k
¢
K__inference_sequential_57_layer_call_and_return_conditional_losses_15196026

inputs/
+dense_627_mlcmatmul_readvariableop_resource-
)dense_627_biasadd_readvariableop_resource/
+dense_628_mlcmatmul_readvariableop_resource-
)dense_628_biasadd_readvariableop_resource/
+dense_629_mlcmatmul_readvariableop_resource-
)dense_629_biasadd_readvariableop_resource/
+dense_630_mlcmatmul_readvariableop_resource-
)dense_630_biasadd_readvariableop_resource/
+dense_631_mlcmatmul_readvariableop_resource-
)dense_631_biasadd_readvariableop_resource/
+dense_632_mlcmatmul_readvariableop_resource-
)dense_632_biasadd_readvariableop_resource/
+dense_633_mlcmatmul_readvariableop_resource-
)dense_633_biasadd_readvariableop_resource/
+dense_634_mlcmatmul_readvariableop_resource-
)dense_634_biasadd_readvariableop_resource/
+dense_635_mlcmatmul_readvariableop_resource-
)dense_635_biasadd_readvariableop_resource/
+dense_636_mlcmatmul_readvariableop_resource-
)dense_636_biasadd_readvariableop_resource/
+dense_637_mlcmatmul_readvariableop_resource-
)dense_637_biasadd_readvariableop_resource
identity¢ dense_627/BiasAdd/ReadVariableOp¢"dense_627/MLCMatMul/ReadVariableOp¢ dense_628/BiasAdd/ReadVariableOp¢"dense_628/MLCMatMul/ReadVariableOp¢ dense_629/BiasAdd/ReadVariableOp¢"dense_629/MLCMatMul/ReadVariableOp¢ dense_630/BiasAdd/ReadVariableOp¢"dense_630/MLCMatMul/ReadVariableOp¢ dense_631/BiasAdd/ReadVariableOp¢"dense_631/MLCMatMul/ReadVariableOp¢ dense_632/BiasAdd/ReadVariableOp¢"dense_632/MLCMatMul/ReadVariableOp¢ dense_633/BiasAdd/ReadVariableOp¢"dense_633/MLCMatMul/ReadVariableOp¢ dense_634/BiasAdd/ReadVariableOp¢"dense_634/MLCMatMul/ReadVariableOp¢ dense_635/BiasAdd/ReadVariableOp¢"dense_635/MLCMatMul/ReadVariableOp¢ dense_636/BiasAdd/ReadVariableOp¢"dense_636/MLCMatMul/ReadVariableOp¢ dense_637/BiasAdd/ReadVariableOp¢"dense_637/MLCMatMul/ReadVariableOp´
"dense_627/MLCMatMul/ReadVariableOpReadVariableOp+dense_627_mlcmatmul_readvariableop_resource*
_output_shapes

:*
dtype02$
"dense_627/MLCMatMul/ReadVariableOp
dense_627/MLCMatMul	MLCMatMulinputs*dense_627/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_627/MLCMatMulª
 dense_627/BiasAdd/ReadVariableOpReadVariableOp)dense_627_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_627/BiasAdd/ReadVariableOp¬
dense_627/BiasAddBiasAdddense_627/MLCMatMul:product:0(dense_627/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_627/BiasAddv
dense_627/ReluReludense_627/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_627/Relu´
"dense_628/MLCMatMul/ReadVariableOpReadVariableOp+dense_628_mlcmatmul_readvariableop_resource*
_output_shapes

:*
dtype02$
"dense_628/MLCMatMul/ReadVariableOp³
dense_628/MLCMatMul	MLCMatMuldense_627/Relu:activations:0*dense_628/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_628/MLCMatMulª
 dense_628/BiasAdd/ReadVariableOpReadVariableOp)dense_628_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_628/BiasAdd/ReadVariableOp¬
dense_628/BiasAddBiasAdddense_628/MLCMatMul:product:0(dense_628/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_628/BiasAddv
dense_628/ReluReludense_628/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_628/Relu´
"dense_629/MLCMatMul/ReadVariableOpReadVariableOp+dense_629_mlcmatmul_readvariableop_resource*
_output_shapes

:*
dtype02$
"dense_629/MLCMatMul/ReadVariableOp³
dense_629/MLCMatMul	MLCMatMuldense_628/Relu:activations:0*dense_629/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_629/MLCMatMulª
 dense_629/BiasAdd/ReadVariableOpReadVariableOp)dense_629_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_629/BiasAdd/ReadVariableOp¬
dense_629/BiasAddBiasAdddense_629/MLCMatMul:product:0(dense_629/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_629/BiasAddv
dense_629/ReluReludense_629/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_629/Relu´
"dense_630/MLCMatMul/ReadVariableOpReadVariableOp+dense_630_mlcmatmul_readvariableop_resource*
_output_shapes

:*
dtype02$
"dense_630/MLCMatMul/ReadVariableOp³
dense_630/MLCMatMul	MLCMatMuldense_629/Relu:activations:0*dense_630/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_630/MLCMatMulª
 dense_630/BiasAdd/ReadVariableOpReadVariableOp)dense_630_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_630/BiasAdd/ReadVariableOp¬
dense_630/BiasAddBiasAdddense_630/MLCMatMul:product:0(dense_630/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_630/BiasAddv
dense_630/ReluReludense_630/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_630/Relu´
"dense_631/MLCMatMul/ReadVariableOpReadVariableOp+dense_631_mlcmatmul_readvariableop_resource*
_output_shapes

:*
dtype02$
"dense_631/MLCMatMul/ReadVariableOp³
dense_631/MLCMatMul	MLCMatMuldense_630/Relu:activations:0*dense_631/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_631/MLCMatMulª
 dense_631/BiasAdd/ReadVariableOpReadVariableOp)dense_631_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_631/BiasAdd/ReadVariableOp¬
dense_631/BiasAddBiasAdddense_631/MLCMatMul:product:0(dense_631/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_631/BiasAddv
dense_631/ReluReludense_631/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_631/Relu´
"dense_632/MLCMatMul/ReadVariableOpReadVariableOp+dense_632_mlcmatmul_readvariableop_resource*
_output_shapes

:*
dtype02$
"dense_632/MLCMatMul/ReadVariableOp³
dense_632/MLCMatMul	MLCMatMuldense_631/Relu:activations:0*dense_632/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_632/MLCMatMulª
 dense_632/BiasAdd/ReadVariableOpReadVariableOp)dense_632_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_632/BiasAdd/ReadVariableOp¬
dense_632/BiasAddBiasAdddense_632/MLCMatMul:product:0(dense_632/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_632/BiasAddv
dense_632/ReluReludense_632/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_632/Relu´
"dense_633/MLCMatMul/ReadVariableOpReadVariableOp+dense_633_mlcmatmul_readvariableop_resource*
_output_shapes

:*
dtype02$
"dense_633/MLCMatMul/ReadVariableOp³
dense_633/MLCMatMul	MLCMatMuldense_632/Relu:activations:0*dense_633/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_633/MLCMatMulª
 dense_633/BiasAdd/ReadVariableOpReadVariableOp)dense_633_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_633/BiasAdd/ReadVariableOp¬
dense_633/BiasAddBiasAdddense_633/MLCMatMul:product:0(dense_633/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_633/BiasAddv
dense_633/ReluReludense_633/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_633/Relu´
"dense_634/MLCMatMul/ReadVariableOpReadVariableOp+dense_634_mlcmatmul_readvariableop_resource*
_output_shapes

:*
dtype02$
"dense_634/MLCMatMul/ReadVariableOp³
dense_634/MLCMatMul	MLCMatMuldense_633/Relu:activations:0*dense_634/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_634/MLCMatMulª
 dense_634/BiasAdd/ReadVariableOpReadVariableOp)dense_634_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_634/BiasAdd/ReadVariableOp¬
dense_634/BiasAddBiasAdddense_634/MLCMatMul:product:0(dense_634/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_634/BiasAddv
dense_634/ReluReludense_634/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_634/Relu´
"dense_635/MLCMatMul/ReadVariableOpReadVariableOp+dense_635_mlcmatmul_readvariableop_resource*
_output_shapes

:*
dtype02$
"dense_635/MLCMatMul/ReadVariableOp³
dense_635/MLCMatMul	MLCMatMuldense_634/Relu:activations:0*dense_635/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_635/MLCMatMulª
 dense_635/BiasAdd/ReadVariableOpReadVariableOp)dense_635_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_635/BiasAdd/ReadVariableOp¬
dense_635/BiasAddBiasAdddense_635/MLCMatMul:product:0(dense_635/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_635/BiasAddv
dense_635/ReluReludense_635/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_635/Relu´
"dense_636/MLCMatMul/ReadVariableOpReadVariableOp+dense_636_mlcmatmul_readvariableop_resource*
_output_shapes

:*
dtype02$
"dense_636/MLCMatMul/ReadVariableOp³
dense_636/MLCMatMul	MLCMatMuldense_635/Relu:activations:0*dense_636/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_636/MLCMatMulª
 dense_636/BiasAdd/ReadVariableOpReadVariableOp)dense_636_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_636/BiasAdd/ReadVariableOp¬
dense_636/BiasAddBiasAdddense_636/MLCMatMul:product:0(dense_636/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_636/BiasAddv
dense_636/ReluReludense_636/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_636/Relu´
"dense_637/MLCMatMul/ReadVariableOpReadVariableOp+dense_637_mlcmatmul_readvariableop_resource*
_output_shapes

:*
dtype02$
"dense_637/MLCMatMul/ReadVariableOp³
dense_637/MLCMatMul	MLCMatMuldense_636/Relu:activations:0*dense_637/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_637/MLCMatMulª
 dense_637/BiasAdd/ReadVariableOpReadVariableOp)dense_637_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_637/BiasAdd/ReadVariableOp¬
dense_637/BiasAddBiasAdddense_637/MLCMatMul:product:0(dense_637/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_637/BiasAdd
IdentityIdentitydense_637/BiasAdd:output:0!^dense_627/BiasAdd/ReadVariableOp#^dense_627/MLCMatMul/ReadVariableOp!^dense_628/BiasAdd/ReadVariableOp#^dense_628/MLCMatMul/ReadVariableOp!^dense_629/BiasAdd/ReadVariableOp#^dense_629/MLCMatMul/ReadVariableOp!^dense_630/BiasAdd/ReadVariableOp#^dense_630/MLCMatMul/ReadVariableOp!^dense_631/BiasAdd/ReadVariableOp#^dense_631/MLCMatMul/ReadVariableOp!^dense_632/BiasAdd/ReadVariableOp#^dense_632/MLCMatMul/ReadVariableOp!^dense_633/BiasAdd/ReadVariableOp#^dense_633/MLCMatMul/ReadVariableOp!^dense_634/BiasAdd/ReadVariableOp#^dense_634/MLCMatMul/ReadVariableOp!^dense_635/BiasAdd/ReadVariableOp#^dense_635/MLCMatMul/ReadVariableOp!^dense_636/BiasAdd/ReadVariableOp#^dense_636/MLCMatMul/ReadVariableOp!^dense_637/BiasAdd/ReadVariableOp#^dense_637/MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::2D
 dense_627/BiasAdd/ReadVariableOp dense_627/BiasAdd/ReadVariableOp2H
"dense_627/MLCMatMul/ReadVariableOp"dense_627/MLCMatMul/ReadVariableOp2D
 dense_628/BiasAdd/ReadVariableOp dense_628/BiasAdd/ReadVariableOp2H
"dense_628/MLCMatMul/ReadVariableOp"dense_628/MLCMatMul/ReadVariableOp2D
 dense_629/BiasAdd/ReadVariableOp dense_629/BiasAdd/ReadVariableOp2H
"dense_629/MLCMatMul/ReadVariableOp"dense_629/MLCMatMul/ReadVariableOp2D
 dense_630/BiasAdd/ReadVariableOp dense_630/BiasAdd/ReadVariableOp2H
"dense_630/MLCMatMul/ReadVariableOp"dense_630/MLCMatMul/ReadVariableOp2D
 dense_631/BiasAdd/ReadVariableOp dense_631/BiasAdd/ReadVariableOp2H
"dense_631/MLCMatMul/ReadVariableOp"dense_631/MLCMatMul/ReadVariableOp2D
 dense_632/BiasAdd/ReadVariableOp dense_632/BiasAdd/ReadVariableOp2H
"dense_632/MLCMatMul/ReadVariableOp"dense_632/MLCMatMul/ReadVariableOp2D
 dense_633/BiasAdd/ReadVariableOp dense_633/BiasAdd/ReadVariableOp2H
"dense_633/MLCMatMul/ReadVariableOp"dense_633/MLCMatMul/ReadVariableOp2D
 dense_634/BiasAdd/ReadVariableOp dense_634/BiasAdd/ReadVariableOp2H
"dense_634/MLCMatMul/ReadVariableOp"dense_634/MLCMatMul/ReadVariableOp2D
 dense_635/BiasAdd/ReadVariableOp dense_635/BiasAdd/ReadVariableOp2H
"dense_635/MLCMatMul/ReadVariableOp"dense_635/MLCMatMul/ReadVariableOp2D
 dense_636/BiasAdd/ReadVariableOp dense_636/BiasAdd/ReadVariableOp2H
"dense_636/MLCMatMul/ReadVariableOp"dense_636/MLCMatMul/ReadVariableOp2D
 dense_637/BiasAdd/ReadVariableOp dense_637/BiasAdd/ReadVariableOp2H
"dense_637/MLCMatMul/ReadVariableOp"dense_637/MLCMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


æ
G__inference_dense_636_layer_call_and_return_conditional_losses_15195488

inputs%
!mlcmatmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MLCMatMul/ReadVariableOp
MLCMatMul/ReadVariableOpReadVariableOp!mlcmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MLCMatMul/ReadVariableOp
	MLCMatMul	MLCMatMulinputs MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	MLCMatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMLCMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
MLCMatMul/ReadVariableOpMLCMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Å
0__inference_sequential_57_layer_call_fn_15195807
dense_627_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCalldense_627_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_sequential_57_layer_call_and_return_conditional_losses_151957602
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)
_user_specified_namedense_627_input"±L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*¼
serving_default¨
K
dense_627_input8
!serving_default_dense_627_input:0ÿÿÿÿÿÿÿÿÿ=
	dense_6370
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:³ë
ü^
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
layer_with_weights-5
layer-5
layer_with_weights-6
layer-6
layer_with_weights-7
layer-7
	layer_with_weights-8
	layer-8

layer_with_weights-9

layer-9
layer_with_weights-10
layer-10
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api

signatures
Æ_default_save_signature
Ç__call__
+È&call_and_return_all_conditional_losses"ÿY
_tf_keras_sequentialàY{"class_name": "Sequential", "name": "sequential_57", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_57", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 31]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_627_input"}}, {"class_name": "Dense", "config": {"name": "dense_627", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 31]}, "dtype": "float32", "units": 30, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_628", "trainable": true, "dtype": "float32", "units": 30, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_629", "trainable": true, "dtype": "float32", "units": 30, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_630", "trainable": true, "dtype": "float32", "units": 30, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_631", "trainable": true, "dtype": "float32", "units": 30, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_632", "trainable": true, "dtype": "float32", "units": 30, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_633", "trainable": true, "dtype": "float32", "units": 30, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_634", "trainable": true, "dtype": "float32", "units": 30, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_635", "trainable": true, "dtype": "float32", "units": 30, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_636", "trainable": true, "dtype": "float32", "units": 30, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_637", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 31}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 31]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_57", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 31]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_627_input"}}, {"class_name": "Dense", "config": {"name": "dense_627", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 31]}, "dtype": "float32", "units": 30, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_628", "trainable": true, "dtype": "float32", "units": 30, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_629", "trainable": true, "dtype": "float32", "units": 30, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_630", "trainable": true, "dtype": "float32", "units": 30, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_631", "trainable": true, "dtype": "float32", "units": 30, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_632", "trainable": true, "dtype": "float32", "units": 30, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_633", "trainable": true, "dtype": "float32", "units": 30, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_634", "trainable": true, "dtype": "float32", "units": 30, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_635", "trainable": true, "dtype": "float32", "units": 30, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_636", "trainable": true, "dtype": "float32", "units": 30, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_637", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "mean_squared_error", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
	

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
É__call__
+Ê&call_and_return_all_conditional_losses"Þ
_tf_keras_layerÄ{"class_name": "Dense", "name": "dense_627", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 31]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_627", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 31]}, "dtype": "float32", "units": 30, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 31}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 31]}}


kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
Ë__call__
+Ì&call_and_return_all_conditional_losses"ë
_tf_keras_layerÑ{"class_name": "Dense", "name": "dense_628", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_628", "trainable": true, "dtype": "float32", "units": 30, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 30}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30]}}


kernel
bias
 trainable_variables
!	variables
"regularization_losses
#	keras_api
Í__call__
+Î&call_and_return_all_conditional_losses"ë
_tf_keras_layerÑ{"class_name": "Dense", "name": "dense_629", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_629", "trainable": true, "dtype": "float32", "units": 30, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 30}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30]}}


$kernel
%bias
&trainable_variables
'	variables
(regularization_losses
)	keras_api
Ï__call__
+Ð&call_and_return_all_conditional_losses"ë
_tf_keras_layerÑ{"class_name": "Dense", "name": "dense_630", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_630", "trainable": true, "dtype": "float32", "units": 30, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 30}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30]}}


*kernel
+bias
,trainable_variables
-	variables
.regularization_losses
/	keras_api
Ñ__call__
+Ò&call_and_return_all_conditional_losses"ë
_tf_keras_layerÑ{"class_name": "Dense", "name": "dense_631", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_631", "trainable": true, "dtype": "float32", "units": 30, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 30}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30]}}


0kernel
1bias
2trainable_variables
3	variables
4regularization_losses
5	keras_api
Ó__call__
+Ô&call_and_return_all_conditional_losses"ë
_tf_keras_layerÑ{"class_name": "Dense", "name": "dense_632", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_632", "trainable": true, "dtype": "float32", "units": 30, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 30}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30]}}


6kernel
7bias
8trainable_variables
9	variables
:regularization_losses
;	keras_api
Õ__call__
+Ö&call_and_return_all_conditional_losses"ë
_tf_keras_layerÑ{"class_name": "Dense", "name": "dense_633", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_633", "trainable": true, "dtype": "float32", "units": 30, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 30}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30]}}


<kernel
=bias
>trainable_variables
?	variables
@regularization_losses
A	keras_api
×__call__
+Ø&call_and_return_all_conditional_losses"ë
_tf_keras_layerÑ{"class_name": "Dense", "name": "dense_634", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_634", "trainable": true, "dtype": "float32", "units": 30, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 30}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30]}}


Bkernel
Cbias
Dtrainable_variables
E	variables
Fregularization_losses
G	keras_api
Ù__call__
+Ú&call_and_return_all_conditional_losses"ë
_tf_keras_layerÑ{"class_name": "Dense", "name": "dense_635", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_635", "trainable": true, "dtype": "float32", "units": 30, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 30}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30]}}


Hkernel
Ibias
Jtrainable_variables
K	variables
Lregularization_losses
M	keras_api
Û__call__
+Ü&call_and_return_all_conditional_losses"ë
_tf_keras_layerÑ{"class_name": "Dense", "name": "dense_636", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_636", "trainable": true, "dtype": "float32", "units": 30, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 30}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30]}}


Nkernel
Obias
Ptrainable_variables
Q	variables
Rregularization_losses
S	keras_api
Ý__call__
+Þ&call_and_return_all_conditional_losses"ì
_tf_keras_layerÒ{"class_name": "Dense", "name": "dense_637", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_637", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 30}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30]}}

Titer

Ubeta_1

Vbeta_2
	Wdecay
Xlearning_ratemmmmmm$m %m¡*m¢+m£0m¤1m¥6m¦7m§<m¨=m©BmªCm«Hm¬Im­Nm®Om¯v°v±v²v³v´vµ$v¶%v·*v¸+v¹0vº1v»6v¼7v½<v¾=v¿BvÀCvÁHvÂIvÃNvÄOvÅ"
	optimizer
Æ
0
1
2
3
4
5
$6
%7
*8
+9
010
111
612
713
<14
=15
B16
C17
H18
I19
N20
O21"
trackable_list_wrapper
Æ
0
1
2
3
4
5
$6
%7
*8
+9
010
111
612
713
<14
=15
B16
C17
H18
I19
N20
O21"
trackable_list_wrapper
 "
trackable_list_wrapper
Î
Ynon_trainable_variables
trainable_variables

Zlayers
	variables
[metrics
regularization_losses
\layer_metrics
]layer_regularization_losses
Ç__call__
Æ_default_save_signature
+È&call_and_return_all_conditional_losses
'È"call_and_return_conditional_losses"
_generic_user_object
-
ßserving_default"
signature_map
": 2dense_627/kernel
:2dense_627/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
°
^non_trainable_variables
trainable_variables

_layers
	variables
`metrics
regularization_losses
alayer_metrics
blayer_regularization_losses
É__call__
+Ê&call_and_return_all_conditional_losses
'Ê"call_and_return_conditional_losses"
_generic_user_object
": 2dense_628/kernel
:2dense_628/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
°
cnon_trainable_variables
trainable_variables

dlayers
	variables
emetrics
regularization_losses
flayer_metrics
glayer_regularization_losses
Ë__call__
+Ì&call_and_return_all_conditional_losses
'Ì"call_and_return_conditional_losses"
_generic_user_object
": 2dense_629/kernel
:2dense_629/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
°
hnon_trainable_variables
 trainable_variables

ilayers
!	variables
jmetrics
"regularization_losses
klayer_metrics
llayer_regularization_losses
Í__call__
+Î&call_and_return_all_conditional_losses
'Î"call_and_return_conditional_losses"
_generic_user_object
": 2dense_630/kernel
:2dense_630/bias
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
°
mnon_trainable_variables
&trainable_variables

nlayers
'	variables
ometrics
(regularization_losses
player_metrics
qlayer_regularization_losses
Ï__call__
+Ð&call_and_return_all_conditional_losses
'Ð"call_and_return_conditional_losses"
_generic_user_object
": 2dense_631/kernel
:2dense_631/bias
.
*0
+1"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
°
rnon_trainable_variables
,trainable_variables

slayers
-	variables
tmetrics
.regularization_losses
ulayer_metrics
vlayer_regularization_losses
Ñ__call__
+Ò&call_and_return_all_conditional_losses
'Ò"call_and_return_conditional_losses"
_generic_user_object
": 2dense_632/kernel
:2dense_632/bias
.
00
11"
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
 "
trackable_list_wrapper
°
wnon_trainable_variables
2trainable_variables

xlayers
3	variables
ymetrics
4regularization_losses
zlayer_metrics
{layer_regularization_losses
Ó__call__
+Ô&call_and_return_all_conditional_losses
'Ô"call_and_return_conditional_losses"
_generic_user_object
": 2dense_633/kernel
:2dense_633/bias
.
60
71"
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
 "
trackable_list_wrapper
±
|non_trainable_variables
8trainable_variables

}layers
9	variables
~metrics
:regularization_losses
layer_metrics
 layer_regularization_losses
Õ__call__
+Ö&call_and_return_all_conditional_losses
'Ö"call_and_return_conditional_losses"
_generic_user_object
": 2dense_634/kernel
:2dense_634/bias
.
<0
=1"
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
non_trainable_variables
>trainable_variables
layers
?	variables
metrics
@regularization_losses
layer_metrics
 layer_regularization_losses
×__call__
+Ø&call_and_return_all_conditional_losses
'Ø"call_and_return_conditional_losses"
_generic_user_object
": 2dense_635/kernel
:2dense_635/bias
.
B0
C1"
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
non_trainable_variables
Dtrainable_variables
layers
E	variables
metrics
Fregularization_losses
layer_metrics
 layer_regularization_losses
Ù__call__
+Ú&call_and_return_all_conditional_losses
'Ú"call_and_return_conditional_losses"
_generic_user_object
": 2dense_636/kernel
:2dense_636/bias
.
H0
I1"
trackable_list_wrapper
.
H0
I1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
non_trainable_variables
Jtrainable_variables
layers
K	variables
metrics
Lregularization_losses
layer_metrics
 layer_regularization_losses
Û__call__
+Ü&call_and_return_all_conditional_losses
'Ü"call_and_return_conditional_losses"
_generic_user_object
": 2dense_637/kernel
:2dense_637/bias
.
N0
O1"
trackable_list_wrapper
.
N0
O1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
non_trainable_variables
Ptrainable_variables
layers
Q	variables
metrics
Rregularization_losses
layer_metrics
 layer_regularization_losses
Ý__call__
+Þ&call_and_return_all_conditional_losses
'Þ"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
n
0
1
2
3
4
5
6
7
	8

9
10"
trackable_list_wrapper
(
0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
¿

total

count
	variables
	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
:  (2total
:  (2count
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
':%2Adam/dense_627/kernel/m
!:2Adam/dense_627/bias/m
':%2Adam/dense_628/kernel/m
!:2Adam/dense_628/bias/m
':%2Adam/dense_629/kernel/m
!:2Adam/dense_629/bias/m
':%2Adam/dense_630/kernel/m
!:2Adam/dense_630/bias/m
':%2Adam/dense_631/kernel/m
!:2Adam/dense_631/bias/m
':%2Adam/dense_632/kernel/m
!:2Adam/dense_632/bias/m
':%2Adam/dense_633/kernel/m
!:2Adam/dense_633/bias/m
':%2Adam/dense_634/kernel/m
!:2Adam/dense_634/bias/m
':%2Adam/dense_635/kernel/m
!:2Adam/dense_635/bias/m
':%2Adam/dense_636/kernel/m
!:2Adam/dense_636/bias/m
':%2Adam/dense_637/kernel/m
!:2Adam/dense_637/bias/m
':%2Adam/dense_627/kernel/v
!:2Adam/dense_627/bias/v
':%2Adam/dense_628/kernel/v
!:2Adam/dense_628/bias/v
':%2Adam/dense_629/kernel/v
!:2Adam/dense_629/bias/v
':%2Adam/dense_630/kernel/v
!:2Adam/dense_630/bias/v
':%2Adam/dense_631/kernel/v
!:2Adam/dense_631/bias/v
':%2Adam/dense_632/kernel/v
!:2Adam/dense_632/bias/v
':%2Adam/dense_633/kernel/v
!:2Adam/dense_633/bias/v
':%2Adam/dense_634/kernel/v
!:2Adam/dense_634/bias/v
':%2Adam/dense_635/kernel/v
!:2Adam/dense_635/bias/v
':%2Adam/dense_636/kernel/v
!:2Adam/dense_636/bias/v
':%2Adam/dense_637/kernel/v
!:2Adam/dense_637/bias/v
é2æ
#__inference__wrapped_model_15195230¾
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *.¢+
)&
dense_627_inputÿÿÿÿÿÿÿÿÿ
2
0__inference_sequential_57_layer_call_fn_15196075
0__inference_sequential_57_layer_call_fn_15196124
0__inference_sequential_57_layer_call_fn_15195699
0__inference_sequential_57_layer_call_fn_15195807À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ú2÷
K__inference_sequential_57_layer_call_and_return_conditional_losses_15195590
K__inference_sequential_57_layer_call_and_return_conditional_losses_15195946
K__inference_sequential_57_layer_call_and_return_conditional_losses_15196026
K__inference_sequential_57_layer_call_and_return_conditional_losses_15195531À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ö2Ó
,__inference_dense_627_layer_call_fn_15196144¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_dense_627_layer_call_and_return_conditional_losses_15196135¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ö2Ó
,__inference_dense_628_layer_call_fn_15196164¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_dense_628_layer_call_and_return_conditional_losses_15196155¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ö2Ó
,__inference_dense_629_layer_call_fn_15196184¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_dense_629_layer_call_and_return_conditional_losses_15196175¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ö2Ó
,__inference_dense_630_layer_call_fn_15196204¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_dense_630_layer_call_and_return_conditional_losses_15196195¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ö2Ó
,__inference_dense_631_layer_call_fn_15196224¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_dense_631_layer_call_and_return_conditional_losses_15196215¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ö2Ó
,__inference_dense_632_layer_call_fn_15196244¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_dense_632_layer_call_and_return_conditional_losses_15196235¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ö2Ó
,__inference_dense_633_layer_call_fn_15196264¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_dense_633_layer_call_and_return_conditional_losses_15196255¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ö2Ó
,__inference_dense_634_layer_call_fn_15196284¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_dense_634_layer_call_and_return_conditional_losses_15196275¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ö2Ó
,__inference_dense_635_layer_call_fn_15196304¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_dense_635_layer_call_and_return_conditional_losses_15196295¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ö2Ó
,__inference_dense_636_layer_call_fn_15196324¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_dense_636_layer_call_and_return_conditional_losses_15196315¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ö2Ó
,__inference_dense_637_layer_call_fn_15196343¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_dense_637_layer_call_and_return_conditional_losses_15196334¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÕBÒ
&__inference_signature_wrapper_15195866dense_627_input"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ±
#__inference__wrapped_model_15195230$%*+0167<=BCHINO8¢5
.¢+
)&
dense_627_inputÿÿÿÿÿÿÿÿÿ
ª "5ª2
0
	dense_637# 
	dense_637ÿÿÿÿÿÿÿÿÿ§
G__inference_dense_627_layer_call_and_return_conditional_losses_15196135\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_dense_627_layer_call_fn_15196144O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ§
G__inference_dense_628_layer_call_and_return_conditional_losses_15196155\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_dense_628_layer_call_fn_15196164O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ§
G__inference_dense_629_layer_call_and_return_conditional_losses_15196175\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_dense_629_layer_call_fn_15196184O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ§
G__inference_dense_630_layer_call_and_return_conditional_losses_15196195\$%/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_dense_630_layer_call_fn_15196204O$%/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ§
G__inference_dense_631_layer_call_and_return_conditional_losses_15196215\*+/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_dense_631_layer_call_fn_15196224O*+/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ§
G__inference_dense_632_layer_call_and_return_conditional_losses_15196235\01/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_dense_632_layer_call_fn_15196244O01/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ§
G__inference_dense_633_layer_call_and_return_conditional_losses_15196255\67/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_dense_633_layer_call_fn_15196264O67/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ§
G__inference_dense_634_layer_call_and_return_conditional_losses_15196275\<=/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_dense_634_layer_call_fn_15196284O<=/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ§
G__inference_dense_635_layer_call_and_return_conditional_losses_15196295\BC/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_dense_635_layer_call_fn_15196304OBC/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ§
G__inference_dense_636_layer_call_and_return_conditional_losses_15196315\HI/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_dense_636_layer_call_fn_15196324OHI/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ§
G__inference_dense_637_layer_call_and_return_conditional_losses_15196334\NO/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_dense_637_layer_call_fn_15196343ONO/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿÑ
K__inference_sequential_57_layer_call_and_return_conditional_losses_15195531$%*+0167<=BCHINO@¢=
6¢3
)&
dense_627_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ñ
K__inference_sequential_57_layer_call_and_return_conditional_losses_15195590$%*+0167<=BCHINO@¢=
6¢3
)&
dense_627_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ç
K__inference_sequential_57_layer_call_and_return_conditional_losses_15195946x$%*+0167<=BCHINO7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ç
K__inference_sequential_57_layer_call_and_return_conditional_losses_15196026x$%*+0167<=BCHINO7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¨
0__inference_sequential_57_layer_call_fn_15195699t$%*+0167<=BCHINO@¢=
6¢3
)&
dense_627_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ¨
0__inference_sequential_57_layer_call_fn_15195807t$%*+0167<=BCHINO@¢=
6¢3
)&
dense_627_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
0__inference_sequential_57_layer_call_fn_15196075k$%*+0167<=BCHINO7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
0__inference_sequential_57_layer_call_fn_15196124k$%*+0167<=BCHINO7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿÇ
&__inference_signature_wrapper_15195866$%*+0167<=BCHINOK¢H
¢ 
Aª>
<
dense_627_input)&
dense_627_inputÿÿÿÿÿÿÿÿÿ"5ª2
0
	dense_637# 
	dense_637ÿÿÿÿÿÿÿÿÿ