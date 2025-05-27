import { ReactNode } from "react";

type TelemetryCardProps = {
  label: string;
  value: string;
  unit?: string;
  icon?: ReactNode;
  color?: string;
};

const TelemetryCard = ({ label, value, unit, icon, color = "text-blue-600" }: TelemetryCardProps) => (
  <div className="bg-white p-4 rounded-lg shadow text-sm w-full flex justify-between items-center">
    <div>
      <div className="text-gray-500 mb-1">{label}</div>
      <div className={`text-xl font-semibold ${color}`}>
        {value} {unit && <span className="text-sm">{unit}</span>}
      </div>
    </div>
    <div className="text-2xl">{icon}</div>
  </div>
);

export default TelemetryCard;
