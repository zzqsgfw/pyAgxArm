from typing import Optional

from .......utiles.validator import Validator
from .....msgs.core import MessageAbstract
from .....msgs.nero.default import ArmMsgFeedbackLeaderJointStates
from ...versions.v111.driver import Driver as V111Driver

from .parser import Parser


class Driver(V111Driver):
    """Nero CAN driver for firmware >= v112 (1.12).

    Terminology
    -----------
    `flange`:
    - The mounting face / connection interface on the robotic arm's last link
      (mechanical tool interface).

    Common conventions
    ------------------
    `timeout` (for request/response style APIs):
    - `timeout < 0.0` raises ValueError.
    - `timeout == 0.0`: non-blocking; evaluate readiness once and return
      immediately.
    - `timeout > 0.0`: blocking; poll until ready or timeout expires.

    `joint_index`:
    - `joint_index == 255` means "all joints".

    `set_*` return semantics:
    - Many `set_*` APIs are ACK-only: True means the controller acknowledged the
      request.
      This does not strictly guarantee the setting is already applied.
    - Some `set_*` APIs additionally verify by reading back state; their
      docstrings will mention the verification method if applicable.
    """

    _Parser = Parser

    def set_normal_mode(self):
        """Set the robotic arm to the normal controlled mode (single arm).

        On firmware v112+, the controller does not implement this path; this
        override is a deliberate no-op so callers using older scripts keep
        running without errors.
        """
        return None

    def get_leader_joint_angles(self):
        """Get the leader arm joint angles,
        can be used to control the follower arm.

        Returns
        -------
        MessageAbstract[list[float]] | None
            The joint angles feedback of the leader arm.
            If the joint angles are not available, return None.

        Message
        -------
        `list[float]`: joint angles, unit: rad

        Examples
        --------
        >>> mja = robot.get_leader_joint_angles()
        >>> if mja is not None:
        >>>     print(mja.msg)
        >>>     print(mja.hz, mja.timestamp)
        """
        leader_joint_angles: Optional[
            MessageAbstract[ArmMsgFeedbackLeaderJointStates]
        ] = None
        if getattr(self, "_leader_joint_angles", None) is None:
            self._leader_joint_angles = MessageAbstract(
                msg=list([0.0] * self._JOINT_NUMS),
                msg_type=ArmMsgFeedbackLeaderJointStates.type_,
            )
        if getattr(self._parser, "leader_joint_12", None) is not None:
            leader_joint_angles = self._parser.leader_joint_12
            self._leader_joint_angles.msg[0] = leader_joint_angles.msg.joint_1
            self._leader_joint_angles.msg[1] = leader_joint_angles.msg.joint_2
        if getattr(self._parser, "leader_joint_34", None) is not None:
            leader_joint_angles = self._parser.leader_joint_34
            self._leader_joint_angles.msg[2] = leader_joint_angles.msg.joint_3
            self._leader_joint_angles.msg[3] = leader_joint_angles.msg.joint_4
        if getattr(self._parser, "leader_joint_56", None) is not None:
            leader_joint_angles = self._parser.leader_joint_56
            self._leader_joint_angles.msg[4] = leader_joint_angles.msg.joint_5
            self._leader_joint_angles.msg[5] = leader_joint_angles.msg.joint_6
        if getattr(self._parser, "leader_joint_7", None) is not None:
            leader_joint_angles = self._parser.leader_joint_7
            self._leader_joint_angles.msg[6] = leader_joint_angles.msg.joint_7
        if leader_joint_angles is not None:
            self._leader_joint_angles.timestamp = leader_joint_angles.timestamp
            self._leader_joint_angles.hz = self._ctx.fps.get_fps(
                leader_joint_angles.msg_type)
            if Validator.is_joints(
                self._leader_joint_angles.msg,
                length=self._JOINT_NUMS
            ):
                return self._leader_joint_angles
        return None