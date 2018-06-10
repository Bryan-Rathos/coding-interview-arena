// Implement a Vector class data-structure in C++
// Work in progress ...

class vector {

    int sz;         // the size
    double* elem;   // pointer to the elements
    int space;      // number of elements plus number of free slots

public:
    vector() : sz(0), elem(0), space(0) {}
    explicit vector(int s) :sz(s), elem(new double[sz]), space(s)
    {
        for (int i=0; i<sz; ++i) elem[i]=0; // elements are initialized
    }

    vector(const vector&);              // copy constructor
    vector& operator=(const vector&);   // copy assignment

    ~vector() { delete[] elem;}         // destructor

    double& operator[](int n){ return elem[n]; }    // access
    const double& operator[](int n) const { return elem[n]; }

    int size() const { return sz; }
    int capacity() const { return space; }

    void resize(int newsize);           // growth
    void push_back(double d);
    void reserve(int newalloc);

};

void vector::copy(const vector& arg)
{
    for (int i=0; i<arg.sz; ++i) elem[i] = arg.elem[i];
}

vector::vector(const vector& arg)
    :sz(arg.sz), elem(new double[arg.sz])
{
    copy(arg)
}

vector& vector::operator=(const vector& a)
{
    double* p = new double[a.sz];   // allocate new space
    for(int i=0; i<a.sz; ++i) p[i]=a.elem[i];   // copy elements
    delete[] elem;  // deallocate old space, this is from target vector
    elem = p;       // now reset elem
    sz = a.sz;
    return *this;   // return a self-reference
}
