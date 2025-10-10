from numpy import cos, sin, rad2deg, deg2rad
from .MobileRobotBase import *
from typing import List


class CTMRDiff(CMobileRobot):
    """CTMRDiff은 Two Wheeled mobile Robot differential drive에 대한 클래스이다.

    Args:
        afWheelRad (float): 바퀴 반지름
        afWheelBase (float): 바퀴 간 거리
        afMaxVel (float): 최대 속도
        afMaxAcc (float): 최대 가속도
        afMaxJerk (float): 최대 저크
        afMaxYaw (float, optional): 최대 각속도. Defaults to 0..
    """
    def __init__(self, afWheelRad: float, afWheelBase: float, afMaxVel: float, afMaxAcc: float, afMaxJerk: float,
                 afMaxYaw: float = 0.):

        super(CTMRDiff, self).__init__(EnumDriveType.DIFFERENTIAL, 2, afWheelRad, afWheelBase)
        self.PhyLimits = (afMaxVel, afMaxAcc, afMaxJerk, afMaxYaw)

    def simulate_robot_drive(self, afXi: float, afXf: float, alistVc: List[float], alistOmegaC: List[float], dT: float):
        x_i = afXi
        y_i = afXf

        listXc = [x_i]
        listYc = [y_i]
        x_c = 0.
        y_c = 0.
        for u, v_u in enumerate(alistVc):
            v_x = v_u * cos(alistOmegaC[u])
            v_y = v_u * sin(alistOmegaC[u])
            x_c += v_x * dT
            y_c += v_y * dT
            listXc.append(x_c)
            listYc.append(y_c)

        return listXc, listYc

    def calculate_joint_space_vel_actl(self, afVelCenter: float, afAngVelCenter: float) -> Tuple[float, float]:
        f_omega_r, f_omega_l = self.calculate_joint_space_vel(afVelCenter, afAngVelCenter)
        f_omega_r = f_omega_r * self.fGearRatio
        f_omega_l = f_omega_l * self.fGearRatio

        return f_omega_r, f_omega_l


    def calculate_joint_space_vel(self, afVelCenter: float, afAngVelCenter: float) -> Tuple[float, float]:
        """ Calculates the joint space velocities
        Returns a tuple of the right and left angular velocities in rad/s
        afVelCenter is the Linear velocity at the center of the mobile robot.
        afAngVelCenter is the yaw rate (Change of heading angle)
        """
        f_omega_r = (afVelCenter + (self.WheelBase / 2) * afAngVelCenter) / self.WheelRad
        f_omega_l = (afVelCenter - (self.WheelBase / 2) * afAngVelCenter) / self.WheelRad

        return f_omega_r, f_omega_l

    def calculate_center_vel(self, atupleAngVels: tuple) -> tuple:
        """ calculates the center velocities of the mobile robot.
        Returns a tuple of central linear vel and yaw rate
        atupleAngVels is a tuple of the right and left angular velocities.
        """
        f_omega_r, f_omega_l = atupleAngVels  # extract joint space angular velocities
        f_vel_c = ((f_omega_r + f_omega_l) / 2) * self.WheelRad  # Linear velocity at the center of the mobile robot
        f_omega_c = ((
                             f_omega_r - f_omega_l) / self.WheelBase) * self.WheelRad  # Angular velocity at the center of the mobile robot
        return f_vel_c, f_omega_c

    def calculate_pose(self, atupleAngVels: tuple, afDeltaT: float,
                       aenumFKMethod: EnumFKMethod = EnumFKMethod.EULER) -> Tuple[RobotPose, float, float]:
        """ calculates the forward kinematics of the mobile robot.
        atupleAngVels is a tuple of the right and left angular velocities.
        afDeltaT refers to the sampling time in seconds.
        aenumFKMethod is the dead-reckoning approach. Default is Euler.
        Other approaches include: RUNGE_KUTTA and EXACT.
        Returns the calculated position in the cartesian plane using the class XYPos, Linear and angular velocities
        at the center of the robot.
        This should be called everytime the robot changes position.
        """
        if len(atupleAngVels) != self.WheelNo:
            raise TypeError  # raises a TypeError exception when the fJointsVel is does not contain 2 values

        # Ensure that the selected FK method is valid
        try:
            if not isinstance(aenumFKMethod, EnumFKMethod):
                aenumFKMethod = EnumFKMethod(aenumFKMethod)
        except ValueError:
            aenumFKMethod = EnumFKMethod.EULER

        f_vel_c, f_omega_c = self.calculate_center_vel(atupleAngVels)
        f_prev_theta = self.stCurrPos.THETA  # used for exact integration
        self.stCurrPos.THETA += f_omega_c * afDeltaT  # same for all fk methods

        if EnumFKMethod.EULER == aenumFKMethod:
            f_vel_x = f_vel_c * afDeltaT * cos(self.stCurrPos.THETA)
            f_vel_y = f_vel_c * afDeltaT * sin(self.stCurrPos.THETA)
        elif EnumFKMethod.EXACT == aenumFKMethod and 0 != f_omega_c:  # todo: parametrize range of allowable omega_c
            f_vel_x = (f_vel_c / f_omega_c) * (sin(self.stCurrPos.THETA) - sin(f_prev_theta))
            f_vel_y = ((f_vel_c / f_omega_c) * (cos(self.stCurrPos.THETA) - cos(f_prev_theta))) * -1
        else:  # EnumFKMethod.RUNGE_KUTTA or EXACT but f_omega_c is 0
            f_vel_x = f_vel_c * afDeltaT * cos(self.stCurrPos.THETA + ((f_omega_c * afDeltaT) / 2))
            f_vel_y = f_vel_c * afDeltaT * sin(self.stCurrPos.THETA + ((f_omega_c * afDeltaT) / 2))

        self.stCurrPos.X_POS += f_vel_x
        self.stCurrPos.Y_POS += f_vel_y

        return self.stCurrPos, f_vel_c, f_omega_c


if __name__ == '__main__':
    cTetra = CTMRDiff(0.0752, 0.289, 1.2, 0.4, 0.2)
    print(cTetra.get_curr_pos_t())
    omega_r, omega_l = cTetra.calculate_joint_space_vel(1.2, 0)
    print((omega_r), (omega_l))


    print(cTetra.calculate_center_vel((omega_r, omega_l)))

    print(cTetra.calculate_pose((omega_r, omega_l), 1))
    print(cTetra.get_curr_pos_t())
    print(cTetra.calculate_pose((omega_r, omega_l), 1))
    print(cTetra.get_curr_pos_t())
