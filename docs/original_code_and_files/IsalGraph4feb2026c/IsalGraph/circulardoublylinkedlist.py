from typing import List


class CircularDoublyLinkedList:
    """
    Array-backed circular doubly linked list with integer payloads.

    This data structure represents a circular doubly linked list using
    integer indices into preallocated arrays. It is optimized for:
    - Constant-time insertion and removal
    - Constant-time access to nodes by index
    - Compact memory usage (no per-node Python objects)

    Nodes are represented by integer indices in the range [0, capacity - 1].

    :param capacity: Maximum number of nodes that can exist in the list
    :type capacity: int
    """

    __slots__ = ("_next", "_prev", "_data", "_free", "_size", "_capacity")

    def __init__(self, capacity: int) -> None:
        """
        Initialize the circular doubly linked list.

        Preallocates all internal arrays and initializes the free list.

        :param capacity: Maximum number of nodes allowed
        :type capacity: int
        """
        # Store the maximum capacity
        self._capacity: int = capacity

        # Array storing the index of the next node for each node
        self._next: List[int] = [-1] * capacity

        # Array storing the index of the previous node for each node
        self._prev: List[int] = [-1] * capacity

        # Array storing the integer payload for each node
        self._data: List[int] = [0] * capacity

        # Stack of free node indices (initially all nodes are free)
        self._free: List[int] = list(range(capacity - 1, -1, -1))

        # Current number of active nodes in the list
        self._size: int = 0

    def size(self) -> int:
        """
        Return the number of elements currently in the list.

        :return: Number of active nodes
        :rtype: int
        """
        return self._size

    def capacity(self) -> int:
        """
        Return the maximum capacity of the list.

        :return: Maximum number of nodes
        :rtype: int
        """
        return self._capacity

    def _allocate_node(self) -> int:
        """
        Allocate a node index from the free list.

        :raises RuntimeError: If the list is already full
        :return: Allocated node index
        :rtype: int
        """
        # Ensure there is free space available
        if not self._free:
            raise RuntimeError("CircularDoublyLinkedList is full")

        # Pop and return a free node index
        return self._free.pop()

    def _free_node(self, index: int) -> None:
        """
        Return a node index to the free list.

        :param index: Node index to free
        :type index: int
        """
        # Append the index back to the free stack
        self._free.append(index)

    def insert_after(self, node: int, value: int) -> int:
        """
        Insert a new node with the given value after the specified node.

        If the list is empty, the new node becomes the sole element and
        points to itself in both directions.

        :param node: Index of the node after which to insert
        :type node: int
        :param value: Integer payload for the new node
        :type value: int
        :return: Index of the newly inserted node
        :rtype: int
        """
        # Allocate a new node index
        new_node: int = self._allocate_node()

        # Store the payload value
        self._data[new_node] = value

        # If the list is empty, initialize circular links
        if self._size == 0:
            self._next[new_node] = new_node
            self._prev[new_node] = new_node
        else:
            # Identify the successor of the given node
            next_node: int = self._next[node]

            # Link current node to new node
            self._next[node] = new_node
            self._prev[new_node] = node

            # Link new node to successor
            self._next[new_node] = next_node
            self._prev[next_node] = new_node

        # Increase the size counter
        self._size += 1

        # Return the index of the new node
        return new_node

    def remove(self, node: int) -> None:
        """
        Remove the specified node from the list.

        :param node: Index of the node to remove
        :type node: int
        """
        # If the list is empty, do nothing
        if self._size == 0:
            return

        # If removing the only node, reset the list
        if self._size == 1:
            self._free_node(node)
            self._size = 0
            return

        # Identify neighboring nodes
        prev_node: int = self._prev[node]
        next_node: int = self._next[node]

        # Bypass the node being removed
        self._next[prev_node] = next_node
        self._prev[next_node] = prev_node

        # Return the node index to the free list
        self._free_node(node)

        # Decrease the size counter
        self._size -= 1

    def get_value(self, node: int) -> int:
        """
        Retrieve the integer payload stored in a node.

        :param node: Node index
        :type node: int
        :return: Stored integer value
        :rtype: int
        """
        return self._data[node]

    def set_value(self, node: int, value: int) -> None:
        """
        Update the integer payload stored in a node.

        :param node: Node index
        :type node: int
        :param value: New integer value
        :type value: int
        """
        self._data[node] = value

    def next_node(self, node: int) -> int:
        """
        Return the index of the next node in the list.

        :param node: Node index
        :type node: int
        :return: Next node index
        :rtype: int
        """
        return self._next[node]

    def prev_node(self, node: int) -> int:
        """
        Return the index of the previous node in the list.

        :param node: Node index
        :type node: int
        :return: Previous node index
        :rtype: int
        """
        return self._prev[node]

    def __str__(self) -> str:
        """
        Return a string representation of the list.
        :return:
        A string representation of the list.
        :rtype: str
        """
        return f"{self.__class__.__name__}(capacity={self._capacity}, size={self._size}, next={self._next}, prev={self._prev}, data={self._data})"




