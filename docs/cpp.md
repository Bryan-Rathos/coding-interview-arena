# C++ Concepts

## Copy Constructors
- Defined to take as its argument a reference to the object from which to copy.
```
vector(const vector&);
```
- We pass by **const** reference because we don't want to modify out argument.

## Copy Assignment
- Can also copy by assignment, if not defined, the default assignment is used.
```
vector& vector::operator=(const vector& a);
```

## Shallow vs Deep Copy
- ```Shallow copy``` copies a pointer so that the two pointers refer to the same object.
- ```Deep copy``` copies the content of a pointer so that the two pointers refer to distinct objects.

```
// Example of shallow copy
int* p = new int(77);
int* q = p; // copy the pointer p
*p = 88;    // change the value
```

```
// Example of deep copy
int* p = new int(77);
int* q = new int(p*);  // allocate a new int then copy the value of p
*p = 88;               // change the value of p, but not q
```

## Templates
- A mechanism to use types as parameters for a class or a function.
- The compiler then generates a specific class of function when we later provide specific types as arguments.
- Allow function and classes to have generic type. This allows a function or a class to work on many different data types without being rewritten for each one.
- A class template can also be called a ```type generator```.

```
template <class identifier> function_declaration;
template <typename identifier> function_declaration;
```

```
template <class myType>
myType GetMax <myType a, myType b> {
    return (a > b ? a:b)
}
// Call this function
k = GetMax<int>(i, j);
```

## Abstract Classes
- An abstract class is, conceptually, a class that cannot be instantiated.
- Implemented as a class that has one or more pure virtual (abstract) functions.
- Intended to be inherited from by concrete classes.

## Virtual Functions
- Without *virtual* you get early binding. In the case of inheritance, which implementation of the method is used gets decided at compile time.
- With *virtual* you get late binding, which implementation of the method gets decided at run time.

### Virtual Destructors
- Allows deleting a derived class object using a pointer to a base class (otherwise results to undefined behaviour)
- Making the base class destructor virtual guarantees that the object of the derived class is destructed properly.

As a rule of thumb: if you have a class with a **virtual** function, it needs a **virtual** destructor.
- If a class has a **virtual** function it is likely to be used as a base class
- If it is a base class its derived class is likely to be allocated using **new**
- If a derived class object is allocated using **new** and manipulated through a pointer to its base
    - Then it is likely to be **deleted** through a pointer to its base.

### Virtual Base Class
- Used in virtual inheritance, a way of preventing multiple instances of a given class appearing in an inheritance hierarchy.

```
class A { public: void Foo() {} };
class B : public A {};
class C : public A {};
class D : public B, public C {};
```

```
  A
 / \
B   C
 \ /
  D
```
```
D d;
d.Foo(); // is this B's Foo() or C's Foo() ??
```

```
class A { public: void Foo() {} };
class B : public virtual A {};
class C : public virtual A {};
class D : public B, public C {};
```
This means that there is only one ``instance`` of A included in the hierarchy.

```
D d;
d.Foo(); // no longer ambiguous
```

## Pointer vs Reference
- A variable which stores the address of another variable is called a pointer.
- Think of a reference as an automatically dereferenced immutable pointer or as an alternative name for an object.
- A pointer has its own memory address and size on the stack (4 bytes on x86) whereas a reference shares the same memory address as the original variable. Can think of reference as another name for that variable.
- Assignment to a pointer changes the pointer's value (not the pointed-to value).
- To get a pointer you need to use **new** or **&**.
    - A pointer needs to be dereferenced with * to access the memory location it points to whereas a reference can be used directly. A pointer to a class/struct uses `->` to access its members whereas a reference uses `.`.
- To access an object pointed to by a pointer you use **\*** or **[]**.
- Assignment to a reference changes the value of the object referred to (not the reference itself)
- You cannot make a reference to a different object after initialization, i.e. a pointer can be reassigned, a reference cannot and must be assigned at initialization.
- You can have pointers to pointers to pointers offering levels of indirection. Whereas references only offer one level of indirection.
- Pointers can be assigned *nullptr* directly, reference cannot. (Beware of null pointers)
- Assignment of references does deep copy; assignment of pointers does not.

## Const pointers
- Read backwards
    - `int*` - point to int
    - `int const *` - pointer to constant int
    - `int * const` - const pointer to int
    - `int const * const` - const pointer to const int
    - `int ** const` - const pointer to a pointer to an int

- Examples
```
const int a = 10;
const int* ptr = &a;
*ptr = 5 // WRONG

```
```
int a = 10;
int* const ptr = &a;
*ptr = 5 // OK

```

## Dereference operator (*)
- When you want to access the value in the memory that the pointer points to
then you *dereference* the pointer.
- The operator itself can be read as ``value pointed to by``.

## Address-of Operator (&)
- The address of a variable can be obtained by preceding the name of a variable
with an ampersand sign (&).
- & is the address-of operator, and can be read simply as "address of"

## Memory allocation using new op
- The ``new`` operator returns a pointer to the allocated memory.
- A pointer value is the address of the first byte of the memory.
- A pointer points to an object of a specified type.
- A pointer does not know how many elements it points to.
- Remember: Allowing assignment of pointers to different types would allow type error.

## Deallocation
- ``delete p`` frees the memory for an individual object allocated by ``new``.
- ``delete[] p`` frees the memory for an array of objects.

## Mutable
- Used to allow a particular data member of const object to be modified.
- So marking specific attribute as modifiable from within `const`.

## C++ inline functions
- An optimization technique used by the compilers to reduce the execution time
- Functions can be instructed to compiler to make them inline so that compiler can
replace those function definition wherever those are being called.
- Compiler replaces the definition of inline functions at compile time instead of referring
function definition at runtime.
- This is just a suggestion to compiler to make the function inline, request may be ignored.

## Static member variables
- Static variables keep their values and are not destroyed even after they go out of scope.
- When we instantiate a class object, each object gets its own copy of all normal member variables.
- Unlike normal member variables, static member variables are shared by all objects of the class.
- Although you can access static members through objects of the class, it turns out that static members
exist even if no objects of the class have been instantiated!
- Think of static members as belonging to the class itself, not to the objects of the class.

## Static member functions
- If the static member variables are public, we can access them directly using the class name and
the scope resolution operator. But if the static member variables are private, static member functions
can be used to access that private member variable.

## Generic Programming
- Writing code that works with a variety of types presented as arguments.
- ```class template``` when we parameterize a class, this is often called ```parameterized type``` or
```parameterized class```.
- ```parameterized function``` when we parameterize a function.
- This form of generic programming relying on explicit template parameters is called ```parametric polymorphism```.
- In contrast, the polymorphism you get from using class heirarchies and virtual functions is called
```ad hoc polymorphism``` and that style of programming is called ```object-oriented programming```.
- There are several differences between object-oriented programming (using class heirarchies and virtual functions) and
generic programming (using templates).
    - Most obvious is that the choice of function invoked when you use generic programming is determined by the
    compiler at compile time; whereas for object-oriented programming, it is not determined until run time.
```
my_vector.push_back(x);     // put x into a vector
circle.draw();              // draw the shape circle
```
- In the examples above, for ```push_back``` the compiler will determine the element type but for ```draw``` the
compiler will indirectly call some ```draw()``` function.
- To sum up:
    - Generic programming: supported by templates, relying on compile-time resolution
    - OOP: supported by class hierarchies and virtual functions, relying on run-time resolution.



