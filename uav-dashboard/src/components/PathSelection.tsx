import React, { useState } from "react";

interface PathSelectionProps {
  onSubmit: (
    start: [number, number, number],
    destination: [number, number, number]
  ) => void;
}

const PathSelection: React.FC<PathSelectionProps> = ({ onSubmit }) => {
  const [start, setStart] = useState<[number, number, number]>([0, 0, 0]);
  const [destination, setDestination] = useState<[number, number, number]>([
    10, 10, 10,
  ]);

  const handleSubmit = () => {
    onSubmit(start, destination);
  };

  return (
    <div className="bg-white p-4 rounded shadow-md">
      <h2 className="text-lg font-semibold mb-4">
        Select Start and Destination
      </h2>
      <div className="mb-4">
        <label className="block text-sm font-medium text-gray-700">
          Start Point:
        </label>
        <div className="flex space-x-2">
          <input
            type="number"
            placeholder="X"
            value={start[0]}
            onChange={(e) => setStart([+e.target.value, start[1], start[2]])}
            className="w-full border rounded px-2 py-1"
          />
          <input
            type="number"
            placeholder="Y"
            value={start[1]}
            onChange={(e) => setStart([start[0], +e.target.value, start[2]])}
            className="w-full border rounded px-2 py-1"
          />
          <input
            type="number"
            placeholder="Z"
            value={start[2]}
            onChange={(e) => setStart([start[0], start[1], +e.target.value])}
            className="w-full border rounded px-2 py-1"
          />
        </div>
      </div>
      <div className="mb-4">
        <label className="block text-sm font-medium text-gray-700">
          Destination Point:
        </label>
        <div className="flex space-x-2">
          <input
            type="number"
            placeholder="X"
            value={destination[0]}
            onChange={(e) =>
              setDestination([+e.target.value, destination[1], destination[2]])
            }
            className="w-full border rounded px-2 py-1"
          />
          <input
            type="number"
            placeholder="Y"
            value={destination[1]}
            onChange={(e) =>
              setDestination([destination[0], +e.target.value, destination[2]])
            }
            className="w-full border rounded px-2 py-1"
          />
          <input
            type="number"
            placeholder="Z"
            value={destination[2]}
            onChange={(e) =>
              setDestination([destination[0], destination[1], +e.target.value])
            }
            className="w-full border rounded px-2 py-1"
          />
        </div>
      </div>
      <button
        onClick={handleSubmit}
        className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
      >
        Simulate
      </button>
    </div>
  );
};

export default PathSelection;
