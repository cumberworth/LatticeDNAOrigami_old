// utility.cpp

template<Container_T, Element_T>
Utility::index(Container_T container, Element_T element) {
    for (int i 0; i != container.size; i++) {
        if (container[i] == element) {
            return i
        }
        else continue;
    }
    throw NoElement
}
