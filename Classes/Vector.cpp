#include <iostream>
#include <VECTOR_PRINT_MODE_ENUM.cpp>

using namespace std;

template <typename datatype>
class Vector
{
private:
	int size = 0;
	datatype *data;

protected:
	bool initiated = true;
	int c_PRINT_MODE = VECTOR_PRINT_MODE::VECTOR_PRINT_MODE_DATA;

private:
	void is_initiated()
	{
		if (!initiated)
			throw std::invalid_argument("Vector is not initialized");
	}
	void is_valid_operation(const Vector &other)
	{
		if (other.get_size() != size)
		{
			throw std::invalid_argument("Vector sizes do not match");
		}
	}

public:
	Vector()
	{
		data = new datatype[0];
		initiated = false;
	}
	Vector(int size) : size(size)
	{
		data = new datatype[size];
		for (int i = 0; i < size; i++)
		{
			data[i] = 0;
		}
	}
	Vector(const Vector &other) : size(other.size)
	{
		data = new datatype[size];
		for (int i = 0; i < size; i++)
		{
			data[i] = other.data[i];
		}
	}
	~Vector()
	{
		delete[] data;
	}
	Vector &operator=(const Vector &other)
	{
		is_initiated();
		if (this != &other)
		{
			delete[] data;
			size = other.size;
			data = new datatype[size];
			for (int i = 0; i < size; i++)
			{
				data[i] = other.data[i];
			}
		}
		return *this;
	}
	datatype &operator[](int index)
	{
		is_initiated();
		if (index < 0 || index >= size)
		{
			throw std::out_of_range("Index out of bounds");
		}
		return data[index];
	}
	Vector operator+(const Vector &other)
	{
		is_initiated();
		is_valid_operation(other);
		Vector result = Vector(size);
		for (int i = 0; i < size; i++)
		{
			result.data[i] = data[i] + other.data[i];
		}
		return result;
	}
	Vector operator* (const Vector& other) {
		is_initiated();
		is_valid_operation(other);
		Vector result = Vector(size);
		for (int i = 0; i < size; i++)
		{
			result.data[i] = data[i] + other.data[i];
		}
		return result;
	}
	Vector operator-(const Vector &other)
	{
		is_initiated();
		is_valid_operation(other);
		Vector result = Vector(size);
		for (int i = 0; i < size; i++)
		{
			result.data[i] = data[i] * other.data[i];
		}
		return result;
	}
	Vector operator*(int scaler)
	{
		is_initiated();
		Vector result = Vector(size);
		for (int i = 0; i < size; i++)
		{
			result.data[i] = data[i] * scaler;
		}
		return result;
	}
	void set_size(int size)
	{
		if (initiated)
		{
			delete[] data;
		}
		this->size = size;
		data = new datatype[size];
		for (int i = 0; i < size; i++)
		{
			data[i] = 0;
		}
		initiated = true;
	}
	int get_size() const
	{
		return size;
	}
	void set_print_mode(VECTOR_PRINT_MODE mode)
	{
		this->c_PRINT_MODE = mode;
	}
	template <typename T>
	friend std::ostream &operator<<(std::ostream &out, const Vector<T> &v);
};

template <typename datatype>
std::ostream &operator<<(std::ostream &out, const Vector<datatype> &v)
{
	if (!v.initiated)
		throw std::invalid_argument("Vector is not initialized");
	if (v.c_PRINT_MODE == VECTOR_PRINT_MODE::VECTOR_PRINT_MODE_NONE)
	{
		cout << "c_PRINT_MODE for Vector is set to NONE" << endl;
		return out;
	}
	if (v.c_PRINT_MODE == VECTOR_PRINT_MODE::VECTOR_PRINT_MODE_FULL || v.c_PRINT_MODE == VECTOR_PRINT_MODE::VECTOR_PRINT_MODE_SIZE)
	{
		out << "Vector Size: " << v.get_size() << std::endl;
		out << "Vector Elements: ";
	}
	if (v.c_PRINT_MODE == VECTOR_PRINT_MODE::VECTOR_PRINT_MODE_FULL || v.c_PRINT_MODE == VECTOR_PRINT_MODE::VECTOR_PRINT_MODE_DATA)
	{
		for (int i = 0; i < v.get_size(); i++)
		{
			out << v.data[i] << " ";
		}
	}
	out << endl;
	return out;
}