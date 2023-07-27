import sys
from typing import Any

sys.path.append("../")
from utils.challenge_2023 import ChallengeSampling

challenge_sampling = ChallengeSampling(noise=True)


"""
####################################
add codes here
####################################
"""

from quri_parts.openfermion.operator import operator_from_openfermion_op
from openfermion.transforms import jordan_wigner
from openfermion.utils import load_operator

from quri_parts.algo.ansatz import HardwareEfficientReal
from quri_parts.algo.optimizer import Adam, OptimizerStatus, LBFGS
from quri_parts.circuit import LinearMappedUnboundParametricQuantumCircuit
from quri_parts.core.measurement import bitwise_commuting_pauli_measurement
from quri_parts.core.sampling.shots_allocator import (
    create_equipartition_shots_allocator,
)
from scipy.linalg import eigh # just for subspace diagonalisation :)
import numpy as np

from quri_parts.core.operator import PAULI_IDENTITY
from quri_parts.core.operator.operator import Operator
from quri_parts.core.operator import pauli_product
from quri_parts.core.operator import pauli_label
   



def cost_fn(hamiltonian, parametric_state, param_values, estimator):
    estimate = estimator(hamiltonian, parametric_state, [param_values])
    return estimate[0].value.real


def vqe(hamiltonian, parametric_state, estimator, init_params, optimizer, num_exec=3):
    opt_state = optimizer.get_init_state(init_params)
    
    def c_fn(param_values):
        return cost_fn(hamiltonian, parametric_state, param_values, estimator)

    def g_fn(param_values):
        grad = parameter_shift_gradient_estimates(
            hamiltonian, parametric_state, param_values, estimator
        )
        return np.asarray([i.real for i in grad.values])

    prev_params = []

    for _ in range(num_exec):

        try:
            opt_state = optimizer.step(opt_state, c_fn, g_fn)
            print(f"iteration {opt_state.niter}")
            if (opt_state.niter%10 == 0 ):
                print(f"qpu_time {challenge_sampling.total_quantum_circuit_time}")
            print(opt_state.cost)
            
        except QuantumCircuitTimeExceededError:
            print("Reached the limit of shots")
            return opt_state, prev_params

        if opt_state.status == OptimizerStatus.FAILED:
            print("Optimizer failed")
            break
        if opt_state.status == OptimizerStatus.CONVERGED:
            print("Optimizer converged")
            break
        
        prev_params.append(opt_state.params)

    return opt_state, prev_params

def translate_and_order(hamiltonian,n_qubits):
    translated_hamiltonian = {}
    for paulilabel,v in hamiltonian.items():
        pw=["I"]*n_qubits
        for i in paulilabel.qubit_indices():
            if(paulilabel.pauli_at(i) == 1):
                pw[i] = "X"
            elif(paulilabel.pauli_at(i) == 2):
                pw[i] = "Y"
            else:
                pw[i] = "Z"
        translated_hamiltonian[(''.join(pw))] = v
        ordered_hamiltonian = dict(sorted(translated_hamiltonian.items(), key=lambda x: np.abs(x[1]), reverse=True))
    return ordered_hamiltonian

def search_basis_element(ordered_hamiltonian):
    # Return pauli word with highest coefficient and a low enough overlap with the identity (and no pauliZ, too instable)
    for pauli in ordered_hamiltonian:
        if(pauli.count('I') < 5 and 'Z' not in pauli):
            return pauli

def expectation_value_H(parametric_state, params_circuit, sampling_estimator,P1,P2, H=Operator({PAULI_IDENTITY: 1.})):

    if(P1 == 'I'*len(P1)):
        p1 = quri_parts.core.operator.pauli.PAULI_IDENTITY
    else:
        s_p1 = ""
        for i,ps in enumerate(P1):
            if(ps != 'I'):
                s_p1 += ps+str(i)+" "
        p1 = pauli_label(s_p1)

    if(P2 == 'I'*len(P2)):
        p2 = quri_parts.core.operator.pauli.PAULI_IDENTITY
    else:
        s_p2 = ""
        for i,ps in enumerate(P2):
            if(ps != 'I'):
                s_p2 += ps+str(i)+" "
        p2 = pauli_label(s_p2)
    
    _H_ = Operator({})
    for pauli, coef in H.items():
        pw,coef1 = pauli_product(p1,pauli)
        pw, coef2 = pauli_product(pw,p2)
        _H_.add_term(pw,coef*coef1*coef2)
    
    return cost_fn(_H_, parametric_state, params_circuit, sampling_estimator) 


# Construct the overlap matrix S
def S_mat(parametric_state, params_circuit, sampling_estimator,basis):
    N = len(basis)
    S_mat = np.ones((N,N), dtype=complex)
    for i in range(N):
        for j in range(i):
            S_mat[i,j] = expectation_value_H(parametric_state, params_circuit, sampling_estimator, basis[i],basis[j])
            S_mat[j,i] = S_mat[i,j].conjugate()
    return S_mat


# Construct the matrix D
def D_mat(parametric_state, params_circuit, sampling_estimator, basis, H, D_mat_old=[]):
    N = len(basis)
    D_mat = np.zeros((N,N), dtype=complex)
    N_old = len(D_mat_old)
    if N_old!=0:
        D_mat[:N_old,:N_old] = D_mat_old.copy()
    for i in range(N_old,N):
        for j in range(i):
            D_mat[i,j] = expectation_value_H(parametric_state, params_circuit, sampling_estimator, basis[i],basis[j], H)
            D_mat[j,i] = D_mat[i,j].conjugate()
        D_mat[i,i] = expectation_value_H(parametric_state, params_circuit, sampling_estimator, basis[i],basis[i], H).real
    
    return D_mat


def Givens(i,j, param, circuit):
    circuit.add_H_gate(i)
    circuit.add_CNOT_gate(i,j)
    circuit.add_ParametricRX_gate(i,param)
    circuit.add_ParametricRX_gate(j,param)
    circuit.add_CNOT_gate(i,j)
    circuit.add_H_gate(i)


class RunAlgorithm:
    def __init__(self) -> None:
        challenge_sampling.reset()

    def result_for_evaluation(self) -> tuple[Any, float]:
        energy_final = self.get_result()
        qc_time_final = challenge_sampling.total_quantum_circuit_time

        return energy_final, qc_time_final

    def get_result(self) -> float:
        """
        ####################################
        add codes here
        ####################################
        """


        # Hamiltonian
        n_site = 4
        n_qubits = 2 * n_site
        ham = load_operator(
            file_name=f"{n_qubits}_qubits_H_5",
            data_directory="../hamiltonian/hamiltonian_samples/",
            plain_text=False,
        )
        jw_hamiltonian = jordan_wigner(ham)
        hamiltonian = operator_from_openfermion_op(jw_hamiltonian)


        # Circuit
        circuit = LinearMappedUnboundParametricQuantumCircuit(n_qubits)
        theta1, theta2, theta3, theta4, theta5, theta6 = circuit.add_parameters("theta1", "theta2", "theta3", "theta4", "theta5", "theta6")
        circuit.add_X_gate(0)
        circuit.add_X_gate(1)
        circuit.add_X_gate(2)
        circuit.add_X_gate(3)
        Givens(2,3, theta1, circuit)
        Givens(2,4, theta2, circuit)
        Givens(2,5, theta3, circuit)
        Givens(3,4, theta4, circuit)
        Givens(3,5, theta5, circuit)
        Givens(4,5, theta6, circuit)
        parametric_state = ParametricCircuitQuantumState(n_qubits, circuit)

        # QPU parameters
        hardware_type = "it"
        shots_allocator = create_equipartition_shots_allocator()
        measurement_factory = bitwise_commuting_pauli_measurement
        n_shots = 750

        sampling_estimator = (
            challenge_sampling.create_concurrent_parametric_sampling_estimator(
                n_shots, measurement_factory, shots_allocator, hardware_type
            )
        )


        # Optimisation
        adam_optimizer = Adam(ftol=10e-5)

        init_param = np.random.rand(circuit.parameter_count) * 2 * np.pi * 0.001

        result, prev_params = vqe(
            hamiltonian,
            parametric_state,
            sampling_estimator,
            init_param,
            adam_optimizer,
        )
        params_circuit = prev_params[-1]

        # Find basis for subspace diagonalisation
        ordered_hamiltonian = translate_and_order(hamiltonian,n_qubits)
        basis_element = search_basis_element(ordered_hamiltonian)
        
        # Subspace diagonalisation

        values_vqe = []
        for _ in range(15):
            values_vqe.append(cost_fn(hamiltonian,parametric_state,params_circuit,sampling_estimator,))
        vqe_val = min(values_vqe)


        basis = ['IIIIIIII',basis_element]
        values = []

        while True:
            print(challenge_sampling.total_quantum_circuit_time)
            try:
                S = S_mat(parametric_state,params_circuit,sampling_estimator,basis)
                D = D_mat(parametric_state,params_circuit,sampling_estimator,basis,hamiltonian)

                vals = eigh(D,S, eigvals_only=True)
                values.append(vals[0])
                
            except QuantumCircuitTimeExceededError:
                break

        eta_plus = 0.075
        eta_moins = 0.075

        values_close = []
        for i in values:
            if i>(1+eta_plus)*vqe_val and i<(1-eta_moins)*vqe_val:
                values_close.append(i)

        if len(values_close)==0:
            return vqe_val
        else:
            return np.mean(values_close)


if __name__ == "__main__":
    run_algorithm = RunAlgorithm()
    print(run_algorithm.get_result())
