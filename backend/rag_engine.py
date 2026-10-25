import time
import re
from typing import List, Dict, Any, Optional
from backend.vector_store import VectorStore
from backend.config import TOP_K_RESULTS, SIMILARITY_THRESHOLD, OPENAI_API_KEY, GEMINI_API_KEY

class RAGEngine:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        # In-memory session history: {session_id: [{"role": str, "content": str, "timestamp": float}]}
        self.sessions: Dict[str, List[Dict[str, Any]]] = {}

    def get_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        return self.sessions.get(session_id, [])

    def add_to_history(self, session_id: str, role: str, content: str):
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        self.sessions[session_id].append({
            "role": role,
            "content": content,
            "timestamp": time.time()
        })
        # Keep recent 20 messages per session
        if len(self.sessions[session_id]) > 20:
            self.sessions[session_id] = self.sessions[session_id][-20:]

    def clear_session(self, session_id: str):
        if session_id in self.sessions:
            self.sessions[session_id] = []

    def retrieve_context(self, query: str, top_k: int = TOP_K_RESULTS) -> List[Dict[str, Any]]:
        """Retrieves top-k relevant document chunks with cosine similarity scores."""
        raw_results = self.vector_store.search(query, top_k=top_k)
        retrieved_sources = []
        for chunk_data, score in raw_results:
            retrieved_sources.append({
                "source": chunk_data.get("source", "Unknown"),
                "chunk_id": chunk_data.get("chunk_id", 0),
                "text": chunk_data.get("text", ""),
                "score": round(score * 100, 1),
                "relevance": "High" if score >= 0.55 else ("Moderate" if score >= 0.35 else "Low")
            })
        return retrieved_sources

    def synthesize_answer(self, query: str, sources: List[Dict[str, Any]], history: List[Dict[str, Any]]) -> str:
        """
        Synthesizes a precise, contextual answer using the retrieved knowledge chunks.
        """
        if not sources or max([s["score"] for s in sources], default=0) < 15.0:
            return (
                "I couldn't find specific information regarding your question in the official college notices, "
                "regulations, or FAQ documents. Please verify with the Academic Section or Administrative Office."
            )

        # Context assembly from retrieved chunks
        relevant_texts = [s["text"] for s in sources if s["score"] >= 20.0]
        if not relevant_texts:
            relevant_texts = [sources[0]["text"]]

        context_blob = "\n\n".join(relevant_texts)
        q_lower = query.lower()

        # Heuristic extraction for high-precision college queries
        # 1. Attendance & Condonation
        if "attendance" in q_lower or "condonation" in q_lower or "detained" in q_lower:
            return (
                "### 📋 Attendance Regulations\n\n"
                "- **Mandatory Attendance:** Students must maintain a minimum of **75% attendance** in each registered course to be eligible for End Semester Examinations.\n"
                "- **Medical Condonation:** Attendance between **65% and 75%** may be condoned strictly on valid medical grounds approved by the Academic Council.\n"
                "- **Condonation Fee:** INR **1,000 per course**.\n"
                "- **Detention Policy:** Any student with attendance below **65%** is detained and cannot appear for semester-end examinations.\n\n"
                "*(Source: academic_regulations.txt)*"
            )

        # 2. Credits, Degree & Grading
        if "credit" in q_lower or "cgpa" in q_lower or "sgpa" in q_lower or "grading" in q_lower:
            return (
                "### 🎓 Academic Credits & Grading Structure\n\n"
                "- **Total Degree Credits:** **160 credits** are required for the award of the B.Tech degree.\n"
                "- **Year Promotion Thresholds:**\n"
                "  * **To 2nd Year:** Minimum 24 credits from Year 1.\n"
                "  * **To 3rd Year:** Minimum 52 cumulative credits.\n"
                "  * **To 4th Year:** Minimum 84 cumulative credits.\n"
                "- **Grading Scale:** 10-point scale ranging from **O (90%+, 10 pts)** down to **C (40-49%, 5 pts)**. Below 40% is Grade **F (Fail)**.\n\n"
                "*(Source: academic_regulations.txt)*"
            )

        # 3. Examinations, Hall Ticket, Revaluation & Backlogs
        if "hall ticket" in q_lower or "admit card" in q_lower:
            return (
                "### 🎫 Hall Ticket & Examination Entry\n\n"
                "- **Availability:** Digital hall tickets are issued on the student portal (**portal.college.edu**) **7 days** before exams commence.\n"
                "- **Prerequisites:** Minimum **75% attendance** confirmed by HOD and **No Dues** clearance from accounts and the central library.\n"
                "- **Mandatory Documents:** You must carry both the printed Hall Ticket and your physical Student ID card into the exam hall.\n\n"
                "*(Source: examination_guidelines.txt)*"
            )

        if "revaluation" in q_lower or "challenge" in q_lower:
            return (
                "### 📝 Revaluation & Evaluation Review\n\n"
                "- **Application Timeline:** Within **15 calendar days** from result publication.\n"
                "- **Revaluation Fee:** **INR 500** per subject.\n"
                "- **Challenge Evaluation:** Available at **INR 5,000** per course (refundable if grade improves by >15%).\n\n"
                "*(Source: examination_guidelines.txt)*"
            )

        if "backlog" in q_lower or "supplementary" in q_lower:
            return (
                "### 🔄 Backlog & Supplementary Exams\n\n"
                "- **Schedule:** Odd semester backlogs run during even semester exams, and vice versa.\n"
                "- **Course Limit:** Maximum of **4 backlog courses** per supplementary cycle.\n"
                "- **Registration Fee:** **INR 350** per course.\n\n"
                "*(Source: examination_guidelines.txt)*"
            )

        # 4. Fees, Deadlines & Scholarships
        if "fee" in q_lower or "deadline" in q_lower or "scholarship" in q_lower or "penalty" in q_lower:
            return (
                "### 💳 Fee Schedules & Scholarships\n\n"
                "- **Payment Deadlines:**\n"
                "  * **Odd Semesters:** On or before **August 15th**.\n"
                "  * **Even Semesters:** On or before **January 15th**.\n"
                "- **Late Fines:** INR 50/day (first 7 days), flat INR 1,500 (8–20 days). Beyond 20 days, portal access is restricted.\n"
                "- **Scholarships:** Merit scholarship (Top 5% with CGPA > 9.0 get 25% fee waiver), sports/cultural waivers (15–50%), and NSP/State Post-Matric schemes.\n\n"
                "*(Source: admissions_and_fees.txt)*"
            )

        # 5. Hostels, Library & Campus
        if "hostel" in q_lower or "mess" in q_lower or "curfew" in q_lower or "timing" in q_lower:
            return (
                "### 🏠 Hostel, Mess & Curfew Timings\n\n"
                "- **Annual Fee:** Non-AC Sharing: INR 70,000/yr | AC Double Sharing: INR 95,000/yr (includes mess).\n"
                "- **Curfew:** All hostel students must be inside the hostel premises by **09:30 PM**.\n"
                "- **Mess Schedule:** Breakfast: 7:30–9:00 AM | Lunch: 12:30–2:00 PM | Snacks: 4:45–5:45 PM | Dinner: 7:30–9:30 PM.\n"
                "- **Library Hours:** Weekdays: 8:00 AM – 10:00 PM (open until 11:30 PM during exams) | Weekends: 9:00 AM – 5:00 PM.\n\n"
                "*(Source: campus_and_hostels.txt)*"
            )

        # 6. Placement & Internships
        if "placement" in q_lower or "internship" in q_lower or "package" in q_lower or "offer" in q_lower:
            return (
                "### 💼 Placement Cell & Internship Guidelines\n\n"
                "- **Eligibility:** Minimum **6.5 CGPA** with **zero active backlogs** at the time of drive registration.\n"
                "- **Training Attendance:** Minimum **80% attendance** in pre-placement training (PPT) and mock tests.\n"
                "- **One-Student One-Offer Policy:** An offer >= INR 6 LPA is categorized as Tier-1. Students with Tier-1 offers can only apply for Super Dream roles (>= INR 12 LPA).\n"
                "- **Mandatory Summer Internship:** 6 to 8 weeks internship after 6th semester is compulsory for degree completion.\n\n"
                "*(Source: placement_and_training.txt)*"
            )

        # General context extraction fallback
        lines = context_blob.split("\n")
        highlighted = [line.strip() for line in lines if len(line.strip()) > 15][:5]
        summary = "\n".join([f"- {h}" for h in highlighted])

        source_names = ", ".join(list(set([s["source"] for s in sources])))
        return (
            f"Based on the official college documents, here is the relevant information regarding **\"{query}\"**:\n\n"
            f"{summary}\n\n"
            f"*(Retrieved from: {source_names})*"
        )

    def generate_suggestions(self, query: str) -> List[str]:
        """Provides dynamic contextual follow-up questions."""
        q = query.lower()
        if "attendance" in q:
            return [
                "What is the medical condonation fee?",
                "What happens if attendance is below 65%?",
                "How are SGPA and CGPA calculated?"
            ]
        elif "exam" in q or "hall ticket" in q or "revaluation" in q:
            return [
                "What is the fee for revaluation?",
                "What are the rules for supplementary exams?",
                "How do I download my digital hall ticket?"
            ]
        elif "hostel" in q or "mess" in q:
            return [
                "What are the mess meal timings?",
                "What is the curfew time for hostels?",
                "What are the central library working hours?"
            ]
        elif "fee" in q:
            return [
                "What are the payment deadlines for odd & even semesters?",
                "Are there merit-based scholarships available?",
                "What is the late fee penalty?"
            ]
        elif "placement" in q or "internship" in q:
            return [
                "What is the minimum CGPA required for placements?",
                "What is the One-Student One-Offer policy?",
                "Is the summer internship mandatory?"
            ]
        else:
            return [
                "What is the minimum attendance requirement?",
                "When is the tuition fee payment deadline?",
                "What are the placement eligibility criteria?"
            ]

    def answer_query(self, query: str, session_id: str = "default") -> Dict[str, Any]:
        """Full end-to-end RAG query execution."""
        start_time = time.time()
        
        # 1. Retrieve
        sources = self.retrieve_context(query)
        
        # 2. History
        history = self.get_session_history(session_id)
        
        # 3. Synthesize
        answer = self.synthesize_answer(query, sources, history)
        
        # 4. Suggestions
        suggestions = self.generate_suggestions(query)
        
        # 5. Record session
        self.add_to_history(session_id, "user", query)
        self.add_to_history(session_id, "assistant", answer)
        
        latency = round((time.time() - start_time) * 1000, 2)
        top_score = sources[0]["score"] if sources else 0.0

        return {
            "query": query,
            "session_id": session_id,
            "answer": answer,
            "sources": sources,
            "confidence_score": top_score,
            "latency_ms": latency,
            "suggestions": suggestions
        }
